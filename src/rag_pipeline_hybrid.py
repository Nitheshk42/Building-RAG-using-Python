from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path
from src.llm_provider import get_llm as _get_llm

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


_RECENCY_WORDS = ("recent", "current", "latest", "currently", "nowadays", "these days")


def _retrieval_query(question):
    """If the question is asking about the recent/current project, bias the retrieval query
    itself toward the terms that mark the newest role (e.g. 'Current', 'Present') so the
    right chunk actually gets retrieved, not just correctly reasoned about after the fact."""
    if any(w in question.lower() for w in _RECENCY_WORDS):
        return question + " Current Present most recent latest role"
    return question


def route_question(question):
    """Classify the question so the user can see WHY each side answered the way it did."""
    llm = _get_llm(temperature=0, max_tokens=120)
    template = """Classify this interview-prep question into exactly one category:
- RESUME_FACT: asking what is literally on the resume (skills, companies, dates)
- TECHNICAL_DEEP_DIVE: asking how/why something was done, challenges, tradeoffs
- BOTH: needs both a resume fact and a technical explanation

Question: {question}

Reply in this exact format:
Category: <RESUME_FACT|TECHNICAL_DEEP_DIVE|BOTH>
Reason: <one short sentence>"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    category, reason = "BOTH", "Defaulted to showing both views."
    for line in result.splitlines():
        if line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return category, reason


def get_resume_chain(vectorstore):
    """Answers strictly from retrieved resume chunks. No elaboration beyond them."""
    llm = _get_llm(temperature=0.2, max_tokens=2000)
    template = """You are answering ONLY using the resume context below. Do not add
outside knowledge or speculation.

SPECIFICITY RULE: If the resume context contains concrete numbers, metrics, config values,
tool versions, or specific project/technology names, quote them verbatim rather than
paraphrasing them away into generic statements.

FORMAT: For narrative/conversational questions ("tell me about yourself," "tell me about your
recent project," etc.) write ONE flowing, coherent answer in first person — do not split into
labeled sections or invent extra sub-topics the question didn't ask about. Only use a per-item
breakdown if the question explicitly enumerates a list of distinct named things to go through
one by one (e.g. "top 10 X", "each of these five Y").

RECENCY RULE: If the question asks about the "recent," "current," "latest," or "most recent"
project/role, do NOT just answer about whichever project happens to appear first in the context
below. Scan ALL project/role entries in the context, compare their date ranges, and identify the
one that is actually most recent — the entry marked "Current" / "Present", or if none is marked
that way, the one with the latest start date. Base the answer on that entry specifically. If the
context doesn't make the dates clear enough to tell, say so rather than guessing.

If the resume context doesn't cover something, say so plainly rather than skipping it.

RESUME CONTEXT:
{context}

QUESTION: {question}

Answer strictly from the resume context:"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(_retrieval_query(x["question"]))),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


LEVEL_INSTRUCTIONS = {
    "Junior": (
        "Explain it the way a JUNIOR engineer would: simpler vocabulary, less discussion of "
        "tradeoffs - but still describe the ACTUAL, SPECIFIC steps taken (which tool did what,"
        " in what order, on what data) using the real details from the resume. 'Simple' means"
        " simple to follow, not vague - never replace real mechanics with a generic summary "
        "like 'we built a pipeline to process data.'"
    ),
    "Mid-Level": (
        "Explain the concrete implementation decisions: which specific tool/service handled "
        "which step, why that tool for that step, and the actual sequence of the pipeline/system "
        "as evidenced in the resume."
    ),
    "Senior": (
        "Go into technical tradeoffs grounded in the specific tools/scale mentioned in the "
        "resume: why alternatives were rejected, edge cases that specific setup would hit, "
        "and how you'd mentor others through it."
    ),
    "Architect": (
        "Frame it at the system level using the specific services/architecture from the "
        "resume: scalability, reliability, cross-team/cross-service concerns, and long-term "
        "tradeoffs of THAT actual setup, not a generic architecture essay."
    ),
}


def get_level_chain(vectorstore, level):
    """Answers the same question calibrated to a specific seniority level."""
    llm = _get_llm(temperature=0.4, max_tokens=1800)
    instruction = LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["Mid-Level"])
    template = """You are helping prepare interview answers based on a resume.

CRITICAL - ACTUAL MECHANICS, NOT SUMMARY: The answer must describe WHAT was actually built and
HOW, step by step, using the real tools/services/data named in the resume context - not a
one-line summary of the outcome. A bad answer says "I built a pipeline to ingest and transform
data." A good answer says specifically which service ingested the data, which tool transformed
it, what format it landed in, and why, using names/numbers straight from the context below.

SPECIFICITY RULE: Quote concrete numbers, metrics, config values, and tool versions from the
resume context verbatim - never paraphrase them into generic statements.

RECENCY RULE: If asked about the "recent," "current," "latest," or "most recent" project/role,
identify the entry marked "Current"/"Present" (or the latest start date) among ALL entries in
the context, and answer about that one specifically - do not default to whichever appears first.

DO NOT relabel the candidate's actual job title or seniority from the resume - if the resume
says "Senior Data Engineer," keep that as the real fact. The Junior/Mid/Senior/Architect level
below only controls HOW MUCH DEPTH and TECHNICAL VOCABULARY to use in explaining it, not what
the person's real title was.

RESUME CONTEXT:
{context}

QUESTION: {question}

""" + instruction + """

Keep the answer focused and interview-ready (not a generic essay), but never sacrifice the
real, specific mechanics for brevity.

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(_retrieval_query(x["question"]))),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


CATEGORY_DEFINITIONS = {
    "Technical": "Technical skills the JD asks for that match the resume.",
    "Behavioral": "Behavioral/experience questions tied to resume projects (teamwork, conflict, ownership).",
    "Resume": (
        "Questions asking the candidate to describe their own project/role/responsibilities "
        "directly from the resume — e.g. 'What was your role in project X?', 'Describe your "
        "responsibilities in Y', 'What was project Z about?'. The answer must be a factual "
        "description pulled straight from the resume context, not general advice."
    ),
    "Gap": "Skills the JD wants that aren't clearly on the resume (still give a good answer strategy).",
}


def check_domain_alignment(vectorstore, jd_text):
    """Quick check: does this JD's core domain actually match what's evidenced in the resume?
    Returns {"aligned": bool, "note": str} so the UI can warn the user before generating
    Q&A that would otherwise falsely imply a fit."""
    llm = _get_llm(temperature=0, max_tokens=150)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    context = format_docs(retriever.invoke(jd_text))
    template = """Compare the core domain/role of this JOB DESCRIPTION against the candidate's
RESUME CONTEXT below. Judge whether the JD's primary domain (e.g. the main tech stack, role
type, or industry it's hiring for) is actually reflected in the resume - not just a shared
buzzword here and there.

JOB DESCRIPTION:
{jd_text}

RESUME CONTEXT:
{context}

Reply in exactly this format:
Aligned: <YES|PARTIAL|NO>
Note: <one short sentence explaining the core domain match or mismatch>"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"jd_text": jd_text, "context": context})

    aligned, note = "PARTIAL", "Could not determine domain fit."
    for line in result.splitlines():
        if line.lower().startswith("aligned:"):
            aligned = line.split(":", 1)[1].strip().upper()
        elif line.lower().startswith("note:"):
            note = line.split(":", 1)[1].strip()
    return {"aligned": aligned, "note": note}


def generate_jd_questions(vectorstore, jd_text, num_questions=8, categories=None, exclude_questions=None):
    """Given a job description, retrieve the most relevant resume chunks and generate
    a structured list of likely interview questions with resume-grounded model answers.

    categories: list of category names to restrict generation to (default: all).
    exclude_questions: list of already-generated question texts, so a "More" click
    doesn't repeat the same questions."""
    categories = categories or list(CATEGORY_DEFINITIONS.keys())
    llm = _get_llm(temperature=0.5, max_tokens=3000)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    context = format_docs(retriever.invoke(jd_text))

    category_desc = "\n".join(f"- {c}: {CATEGORY_DEFINITIONS[c]}" for c in categories)
    avoid_block = ""
    if exclude_questions:
        avoid_list = "\n".join(f"- {q}" for q in exclude_questions)
        avoid_block = f"\nDo NOT repeat or closely rephrase any of these already-asked questions:\n{avoid_list}\n"

    category_options = " | ".join(categories)
    template = """You are an interview coach. Below is a JOB DESCRIPTION and the candidate's
RESUME CONTEXT (retrieved as most relevant to this JD).

JOB DESCRIPTION:
{jd_text}

RESUME CONTEXT:
{context}

HONESTY ABOUT DOMAIN FIT: First, judge whether the JD's core domain (main tech stack, role
type) actually matches what's in the resume. If the resume has genuinely little or no overlap
with the JD's core domain, do NOT pretend a fit — weight questions toward the "Gap" category,
and for any question you do write, the answer must honestly acknowledge what's transferable
(general engineering fundamentals, adjacent tools) rather than inventing direct experience the
resume doesn't show. Never fabricate a project or skill the resume doesn't evidence just because
the JD asks for it.

Generate exactly {num_questions} likely interview questions for this JD, ONLY using these categories:
{category_desc}
{avoid_block}
QUESTION STYLE: Phrase each question the way a real interviewer would say it out loud, naming
the actual company/project/technology from the resume where relevant — e.g. "Walk me through
the Kafka producer you built at Wells Fargo" rather than "Describe your experience with Kafka."

ANSWER STYLE: Write the answer as a flowing first-person narrative (not bullet points), the way
the candidate would actually say it in an interview. Carry over any concrete numbers, config
values, tool versions, or metrics from the resume VERBATIM — never smooth them into vague
phrases. If the resume doesn't have that level of detail for a given point, don't invent it;
keep the answer honest about what's actually there.

For EACH question, output in this EXACT format (use "---" as a separator between questions, no extra text):

Category: <one of: {category_options}>
Question: <the interview question, phrased naturally, naming specifics from the resume>
Answer: <first-person narrative answer, 4-7 sentences, as concrete as the resume allows>
---

Begin now:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "jd_text": jd_text, "context": context, "num_questions": num_questions,
        "category_desc": category_desc, "avoid_block": avoid_block, "category_options": category_options
    })

    items = []
    for block in result.split("---"):
        block = block.strip()
        if not block:
            continue
        category, question, answer = "General", "", ""
        for line in block.splitlines():
            if line.lower().startswith("category:"):
                category = line.split(":", 1)[1].strip()
            elif line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
            elif answer:
                answer += " " + line.strip()
        if question and answer:
            items.append({"category": category, "question": question, "answer": answer})
    return items


def generate_general_jd_questions(jd_text, num_questions=6, exclude_questions=None):
    """Given ONLY a job description - no resume, no retrieval, no candidate-specific context
    at all - generates likely interview questions the way a general-purpose LLM (ChatGPT-style)
    would if you just pasted the JD in and asked for interview prep. For EACH question, answers
    are generated at all four seniority levels (Junior/Mid-Level/Senior/Architect) using generic,
    best-practice domain knowledge - deliberately NOT grounded in anyone's actual resume."""
    llm = _get_llm(temperature=0.6, max_tokens=4000)

    avoid_block = ""
    if exclude_questions:
        avoid_list = "\n".join(f"- {q}" for q in exclude_questions)
        avoid_block = f"\nDo NOT repeat or closely rephrase any of these already-asked questions:\n{avoid_list}\n"

    level_desc = "\n".join(f"- {lvl}: {instr}" for lvl, instr in LEVEL_INSTRUCTIONS.items())

    template = """You are an expert technical interviewer. Below is a JOB DESCRIPTION only -
you have NO candidate resume, NO personal background, nothing specific to any individual.
Generate interview questions and answers purely from general domain expertise for this role,
the same way you'd answer if someone pasted just this JD into a general-purpose AI assistant
and asked for interview prep.

JOB DESCRIPTION:
{jd_text}

Generate exactly {num_questions} likely interview questions for this role, covering a mix of
technical and role-relevant conceptual questions grounded in what the JD actually asks for.
{avoid_block}
For EACH question, write FOUR separate answers - one per seniority level below. Each answer
must be genuinely different in depth and framing, not the same content reworded:
{level_desc}

Answers should read like strong, generic best-practice interview answers - the kind a
well-prepared candidate at that level would give. Do NOT invent a fake personal story, company
name, or "I did X at my last job" claim - keep answers framed around approach, reasoning, and
domain knowledge rather than fabricated personal history.

For EACH question, output in this EXACT format (use "===" as a separator between questions):

Question: <the interview question>
Junior: <junior-level answer, 3-5 sentences>
Mid-Level: <mid-level answer, 3-5 sentences>
Senior: <senior-level answer, 4-6 sentences, tradeoffs/depth>
Architect: <architect-level answer, 4-6 sentences, system-level framing>
===

Begin now:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "jd_text": jd_text, "num_questions": num_questions,
        "avoid_block": avoid_block, "level_desc": level_desc,
    })

    items = []
    for block in result.split("==="):
        block = block.strip()
        if not block:
            continue
        question = ""
        answers = {"Junior": "", "Mid-Level": "", "Senior": "", "Architect": ""}
        current_key = None
        for line in block.splitlines():
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("question:"):
                question = stripped.split(":", 1)[1].strip()
                current_key = None
            elif low.startswith("junior:"):
                answers["Junior"] = stripped.split(":", 1)[1].strip()
                current_key = "Junior"
            elif low.startswith("mid-level:") or low.startswith("mid level:"):
                answers["Mid-Level"] = stripped.split(":", 1)[1].strip()
                current_key = "Mid-Level"
            elif low.startswith("senior:"):
                answers["Senior"] = stripped.split(":", 1)[1].strip()
                current_key = "Senior"
            elif low.startswith("architect:"):
                answers["Architect"] = stripped.split(":", 1)[1].strip()
                current_key = "Architect"
            elif current_key and stripped:
                answers[current_key] += " " + stripped
        if question and any(answers.values()):
            items.append({"question": question, "answers": answers})
    return items


def get_technical_chain(vectorstore):
    """Resume-aware but expands like an interview follow-up: approach, challenges, resolution.
    Must be honest about what the resume actually evidences vs. general knowledge -
    never invent first-person "I did X" claims the resume doesn't support."""
    llm = _get_llm(temperature=0.3, max_tokens=3000)
    template = """You are helping prepare for a technical interview follow-up: "walk me through
how you did that."

CRITICAL HONESTY RULE: Only say "I did X" / "In my role I..." if the RESUME CONTEXT below
actually contains evidence of it. Never invent a specific first-person story, project, or
outcome that isn't in the context. If the resume does NOT clearly cover something, say so
plainly and instead explain how you WOULD approach it in general — do not disguise general
knowledge as personal history.

SPECIFICITY RULE: When the resume context contains concrete details — exact numbers, metrics,
config values, tool versions, timeframes, specific class/service/project names — carry them
into the answer VERBATIM. Do not smooth a specific number into a vague phrase like "improved
performance." If the resume says "tuned thread pool size to 10," say that exact number.

FORMAT DECISION — read the question carefully before choosing:
- If it's a NARRATIVE/CONVERSATIONAL question ("tell me about yourself," "walk me through your
  background," "tell me about yourself and your recent project," etc.) — even if it has multiple
  clauses — write ONE single flowing, well-written narrative answer in first person, the way a
  real candidate would actually speak in an interview. Do NOT split it into labeled sections. Do
  NOT invent extra sub-topics the question didn't ask about. Weave the resume's actual companies,
  roles, tools, and specifics naturally into the story, in a logical order (e.g. current role ->
  what you work on -> a recent project -> how you approach it).
- ONLY if the question explicitly enumerates a list of distinct named items to go through one by
  one (e.g. "top 10 OWASP risks," "walk me through each of these five concepts") should you use a
  separate labeled block per item, in this format:
  ### <item name/number>
  **Your resume evidence:** ... **Approach/Tools:** ... **Challenges & Resolution:** ...

Default to the narrative style unless the question is unambiguously a numbered/enumerated list.

RECENCY RULE: If the question asks about the "recent," "current," "latest," or "most recent"
project/role, do NOT default to whichever project happens to appear first in the context below.
Scan ALL project/role entries in the context, compare their date ranges, and identify the one
that is actually most recent — the entry marked "Current" / "Present", or if none is marked that
way, the one with the latest start date. Base the answer on that entry specifically.

RESUME CONTEXT:
{context}

QUESTION: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(_retrieval_query(x["question"]))),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
