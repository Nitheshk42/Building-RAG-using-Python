from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)


def _get_llm(temperature=0.3, max_tokens=1024):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing! Check .env file exists at project root")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )


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
        "Answer as a JUNIOR engineer would in an interview: keep it simple, focus on "
        "what was built and the basic how, avoid heavy jargon, show willingness to learn."
    ),
    "Mid-Level": (
        "Answer as a MID-LEVEL engineer would in an interview: solid technical detail, "
        "explain the concrete decisions made and common patterns/tools used."
    ),
    "Senior": (
        "Answer as a SENIOR engineer would in an interview: go into technical tradeoffs, "
        "why alternatives were rejected, edge cases, and how you'd mentor others on this."
    ),
    "Architect": (
        "Answer as a SOLUTIONS ARCHITECT would in an interview: focus on system-level design, "
        "scalability, reliability, cross-team/cross-service concerns, and long-term tradeoffs."
    ),
}


def get_level_chain(vectorstore, level):
    """Answers the same question calibrated to a specific seniority level."""
    llm = _get_llm(temperature=0.5, max_tokens=1500)
    instruction = LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["Mid-Level"])
    template = """You are helping prepare interview answers based on a resume.

SPECIFICITY RULE: If the resume context contains concrete numbers, metrics, config values,
or tool versions, quote them verbatim rather than paraphrasing into generic statements —
concrete detail is what makes an answer credible in an interview.

RESUME CONTEXT:
{context}

QUESTION: {question}

""" + instruction + """

Keep the answer focused and interview-ready (not a generic essay), but don't sacrifice
concrete detail for brevity.

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
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
