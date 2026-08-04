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
    """Finds the first two project sections in the resume and, against the JD, produces four
    distinct things per project, kept strictly separate so nothing false gets implied as fact:
    - EDITED: pure truthful rephrasing of what's already in the resume, no new claims
    - SUGGESTIONS: plausible additions inferred from context - NOT auto-applied, user must approve
    - METRIC_RECOMMENDATIONS: placeholder prompts for quantitative numbers (e.g. "reduced X by __%") -
      never invented numbers, always a fill-in-the-blank the user completes with their real figure
    - MISSING_SKILLS: JD-required tools/tech NOT evidenced anywhere in the resume - flagged as a
      learning gap, never phrased as something the candidate already did"""
    llm = _get_llm()
    template = """You are a resume coach. Below is a candidate's RESUME and a JOB DESCRIPTION.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Find the FIRST TWO distinct project/experience entries in the resume (in the order they
appear). For EACH of the two, produce FOUR separate things:

1. ORIGINAL: the verbatim text of that project section, copied exactly from the resume, so it
   can be matched back against the original document.
2. EDITED: a reworded/reordered version that ONLY emphasizes what is ALREADY TRUE in the
   original text, to surface keywords matching the JD. Do NOT add any tool, claim, or number
   that isn't already in the ORIGINAL text. No invented metrics here.
3. SUGGESTIONS: 2-4 bullet points of additional TRUE-SOUNDING points plausible from context and
   the JD's needs, phrased as something to verify before including — never claim a tool/platform
   the candidate has no evidence of using.
4. METRIC_RECOMMENDATIONS: 1-3 bullet points, each a full illustrative bullet point with a
   PLAUSIBLE ESTIMATED RANGE for a quantitative metric this project is missing — e.g. "Reduced
   processing time by 25-30% through this optimization" or "Handled roughly 5,000-10,000
   requests/day at peak load." Base the range on what's typical/reasonable for this kind of
   work, not a wild guess. These are illustrative estimates for the candidate to review, not
   confirmed facts — the candidate must replace them with their real number if different, or
   remove them if they don't apply.
5. MISSING_SKILLS: a bullet list of tools/languages/platforms the JOB DESCRIPTION asks for that
   are NOT evidenced anywhere in this project (or the whole resume). Phrase each as "You don't
   appear to have hands-on experience with X per your resume — if you've genuinely used it,
   add it; otherwise consider this a skill gap to address before applying," never as a claim to
   insert. If none, write "None found."

Respond in EXACTLY this format, nothing else:

PROJECT1_ORIGINAL:
<verbatim text>
PROJECT1_EDITED:
<edited text>
PROJECT1_SUGGESTIONS:
- <suggestion>
PROJECT1_METRIC_RECOMMENDATIONS:
- <illustrative metric bullet with an estimated range>
PROJECT1_MISSING_SKILLS:
- <missing skill note, or "None found">
===
PROJECT2_ORIGINAL:
<verbatim text>
PROJECT2_EDITED:
<edited text>
PROJECT2_SUGGESTIONS:
- <suggestion>
PROJECT2_METRIC_RECOMMENDATIONS:
- <illustrative metric bullet with an estimated range>
PROJECT2_MISSING_SKILLS:
- <missing skill note, or "None found">

Begin now:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"resume_text": resume_text, "jd_text": jd_text})
    return _parse_projects(result)


def _bullets(text):
    return [line.strip("- ").strip() for line in text.strip().splitlines() if line.strip().startswith("-")]


def _parse_projects(raw):
    blocks = raw.split("===")
    projects = []
    pattern = re.compile(
        r"PROJECT\d+_ORIGINAL:\s*(?P<original>.*?)\s*PROJECT\d+_EDITED:\s*(?P<edited>.*?)\s*"
        r"PROJECT\d+_SUGGESTIONS:\s*(?P<suggestions>.*?)\s*"
        r"PROJECT\d+_METRIC_RECOMMENDATIONS:\s*(?P<metrics>.*?)\s*"
        r"PROJECT\d+_MISSING_SKILLS:\s*(?P<missing>.*)",
        re.DOTALL
    )
    for block in blocks:
        match = pattern.search(block)
        if not match:
            continue
        projects.append({
            "original": match.group("original").strip(),
            "edited": match.group("edited").strip(),
            "suggestions": _bullets(match.group("suggestions")),
            "metric_recommendations": _bullets(match.group("metrics")),
            "missing_skills": [s for s in _bullets(match.group("missing")) if s.lower() != "none found"],
        })
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
