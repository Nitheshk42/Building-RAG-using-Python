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

If the question asks about MULTIPLE distinct items (e.g. "top 10 X", "each of Y"),
go through EVERY item the question lists and state for each one, briefly, whether the
resume context covers it and what it says if so.

If the resume context doesn't cover something, say so plainly rather than skipping it.

RESUME CONTEXT:
{context}

QUESTION: {question}

Answer strictly from the resume context:"""
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

Generate exactly {num_questions} likely interview questions for this JD, ONLY using these categories:
{category_desc}
{avoid_block}
For EACH question, output in this EXACT format (use "---" as a separator between questions, no extra text):

Category: <one of: {category_options}>
Question: <the interview question>
Answer: <a strong resume-grounded model answer, 3-5 sentences, specific not generic>
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
    never invent first-person "I did X" claims the resume doesn't support. Must expand
    into one block per item when the question asks about multiple items (e.g. "top 10")."""
    llm = _get_llm(temperature=0.3, max_tokens=3000)
    template = """You are helping prepare for a technical interview follow-up: "walk me through
how you did that."

CRITICAL HONESTY RULE: Only say "I did X" / "In my role I..." if the RESUME CONTEXT below
actually contains evidence of it. Never invent a specific first-person story, project, or
outcome that isn't in the context. If the resume does NOT clearly cover an item, say so
plainly and instead explain how you WOULD approach it in general — do not disguise general
knowledge as personal history.

MULTI-ITEM RULE: If the question asks about several distinct items (e.g. "top 10 OWASP risks",
"each of these concepts"), you MUST cover EVERY item listed, one at a time, each as its own
labeled block below — do not collapse them into a single generic answer.

SPECIFICITY RULE: When the resume context contains concrete details — exact numbers, metrics,
config values, tool versions, thread/pool sizes, throughput figures, timeframes, specific
class/service/project names — you MUST carry them into the answer VERBATIM. Do not smooth a
specific number into a vague phrase like "improved performance." If the resume says "tuned
thread pool size to 10" or "5000 records every 30 seconds," the answer must say that exact
number, not "optimized the configuration."

For EACH item, use this exact format:

### <item name/number>
**Your resume evidence:** what the resume actually shows for this item, quoting concrete
specifics verbatim where present, OR "Not directly evidenced in resume — general approach:"
if it isn't there
**Approach/Tools:** specific tools/technologies/config values (from resume if present, else
typical ones, clearly marked as general knowledge)
**Challenges & Resolution:** the specific failure mode and specific fix if evidenced (e.g. a
concrete bottleneck and the concrete change that resolved it), otherwise phrase as "a common
challenge here would be..."

If this is a SINGLE-item question (not a "top N" list), you may expand each field to 3-5
sentences of concrete, narrative detail instead of 1-2 — depth matters more than brevity here.
Be specific and vary the wording across items — do not reuse the same sentence template.

RESUME CONTEXT:
{context}

QUESTION: {question}

Answer (cover every item asked about):"""
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
