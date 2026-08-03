import re
import difflib
import io
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from docx import Document as DocxDocument


def _get_llm(temperature=0.4, max_tokens=3000):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing! Check .env file exists at project root")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )


def analyze_resume_for_jd(resume_text, jd_text):
    """Finds the first two project sections in the resume and, against the JD, produces:
    - an edited version that only rephrases/emphasizes things that are ALREADY TRUE in the resume
    - a separate list of suggested additions inferred from context, which are NOT auto-applied -
      the user must review and approve each one before it goes into the final document."""
    llm = _get_llm()
    template = """You are a resume coach. Below is a candidate's RESUME and a JOB DESCRIPTION.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Find the FIRST TWO distinct project/experience entries in the resume (in the order they
appear). For EACH of the two, produce:

1. The ORIGINAL text of that project section, copied VERBATIM from the resume (do not alter
   wording, so it can be matched back against the original document).
2. An EDITED version that rewords/reorders/emphasizes ONLY what is already true in the
   original text, to surface keywords and phrasing that match the JD. Do NOT add any claim,
   tool, or outcome that isn't already stated in the ORIGINAL text.
3. SUGGESTIONS: a short bullet list (2-4 items) of additional points that are NOT confirmed by
   the resume text, but are plausible based on the surrounding context and the JD's needs. Each
   one must be phrased as a suggestion to verify, not a fact — the user will decide whether to
   include each one.

Respond in EXACTLY this format, nothing else:

PROJECT1_ORIGINAL:
<verbatim text>
PROJECT1_EDITED:
<edited text>
PROJECT1_SUGGESTIONS:
- <suggestion>
- <suggestion>
===
PROJECT2_ORIGINAL:
<verbatim text>
PROJECT2_EDITED:
<edited text>
PROJECT2_SUGGESTIONS:
- <suggestion>
- <suggestion>

Begin now:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"resume_text": resume_text, "jd_text": jd_text})
    return _parse_projects(result)


def _parse_projects(raw):
    blocks = raw.split("===")
    projects = []
    pattern = re.compile(
        r"PROJECT\d+_ORIGINAL:\s*(?P<original>.*?)\s*PROJECT\d+_EDITED:\s*(?P<edited>.*?)\s*"
        r"PROJECT\d+_SUGGESTIONS:\s*(?P<suggestions>.*)",
        re.DOTALL
    )
    for block in blocks:
        match = pattern.search(block)
        if not match:
            continue
        original = match.group("original").strip()
        edited = match.group("edited").strip()
        suggestions_raw = match.group("suggestions").strip()
        suggestions = [
            line.strip("- ").strip()
            for line in suggestions_raw.splitlines()
            if line.strip().startswith("-")
        ]
        projects.append({"original": original, "edited": edited, "suggestions": suggestions})
    return projects


def render_word_diff(original, edited):
    """GitHub-style inline word diff: red strikethrough for removed, green for added."""
    orig_words = original.split()
    edit_words = edited.split()
    sm = difflib.SequenceMatcher(None, orig_words, edit_words)
    html_parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            html_parts.append(" ".join(orig_words[i1:i2]))
        elif tag == "delete":
            html_parts.append(
                f'<span style="background:#4a1414;color:#ff8a8a;text-decoration:line-through;padding:1px 3px;border-radius:3px;">{" ".join(orig_words[i1:i2])}</span>'
            )
        elif tag == "insert":
            html_parts.append(
                f'<span style="background:#143d1d;color:#8affa0;padding:1px 3px;border-radius:3px;">{" ".join(edit_words[j1:j2])}</span>'
            )
        elif tag == "replace":
            html_parts.append(
                f'<span style="background:#4a1414;color:#ff8a8a;text-decoration:line-through;padding:1px 3px;border-radius:3px;">{" ".join(orig_words[i1:i2])}</span>'
            )
            html_parts.append(
                f'<span style="background:#143d1d;color:#8affa0;padding:1px 3px;border-radius:3px;">{" ".join(edit_words[j1:j2])}</span>'
            )
    return " ".join(html_parts)


def build_tailored_docx(full_resume_text, replacements):
    """replacements: list of (original_snippet, final_text) tuples to substitute into the
    full resume text. Falls back to appending if an exact match isn't found (PDF text
    extraction can introduce whitespace differences)."""
    final_text = full_resume_text
    unmatched = []
    for original_snippet, final_snippet in replacements:
        if original_snippet in final_text:
            final_text = final_text.replace(original_snippet, final_snippet, 1)
        else:
            unmatched.append(final_snippet)

    doc = DocxDocument()
    for line in final_text.split("\n"):
        doc.add_paragraph(line)

    if unmatched:
        doc.add_paragraph("")
        doc.add_paragraph("--- Tailored additions (could not auto-place in original text) ---")
        for snippet in unmatched:
            doc.add_paragraph(snippet)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
