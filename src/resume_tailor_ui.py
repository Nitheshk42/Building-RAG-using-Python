import streamlit as st
from src.resume_tailor import (
    analyze_resume_for_jd, render_side_by_side_diff, build_tailored_docx,
    insert_naturally, render_tailored_preview_markdown,
    extract_jd_keywords, ats_match_score
)


def _ats_score_block(current_text, jd_keywords, baseline_score, container):
    """Renders a live ATS keyword-match score. Recomputes on every rerun (Streamlit reruns the
    whole page on each checkbox click), so checking a suggestion on/off updates the score
    immediately - no extra button needed."""
    score, matched, missing = ats_match_score(current_text, jd_keywords)
    delta = score - baseline_score
    with container:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                "🎯 ATS Match Score",
                f"{score}%",
                delta=f"{delta:+d}%" if delta != 0 else None,
            )
        with col2:
            st.progress(min(score, 100) / 100)
            if missing:
                st.caption(
                    f"🔎 Missing from your resume ({len(missing)}): "
                    + ", ".join(missing[:12]) + ("..." if len(missing) > 12 else "")
                )
            else:
                st.caption("✅ All detected JD keywords are present in your resume.")
    return score


def display_resume_tailor():
    st.title("🎯 Resume Tailor")
    st.markdown(
        "Paste a job description — for your **first two projects**, this suggests new points "
        "you could add to match it. **Nothing changes until you check a box** — your existing "
        "resume text is never rewritten or touched, only added to if you approve it."
    )
    st.divider()

    documents = st.session_state.get("documents")
    if not documents:
        st.warning("⚠️ Please go to Visual tab and process your resume first.")
        return

    full_resume_text = "\n".join(doc.page_content for doc in documents)
    jd_text = st.text_area("Paste the job description here:", height=180, placeholder="Paste JD text...")

    if st.button("🎯 Analyze & Suggest", use_container_width=True):
        if not jd_text.strip():
            st.error("❌ Please paste a job description first.")
            return
        with st.spinner("🔍 Matching your first two projects against this JD..."):
            projects = analyze_resume_for_jd(full_resume_text, jd_text)
        if not projects:
            st.error("❌ Couldn't parse a result — try again.")
            return
        jd_keywords = extract_jd_keywords(jd_text)
        baseline_score, _, _ = ats_match_score(full_resume_text, jd_keywords)
        st.session_state.tailor_projects = projects
        st.session_state.tailor_full_text = full_resume_text
        st.session_state.tailor_approved = [set() for _ in projects]
        st.session_state.tailor_approved_skills = [set() for _ in projects]
        st.session_state.tailor_jd_keywords = jd_keywords
        st.session_state.tailor_baseline_score = baseline_score

    if "tailor_projects" not in st.session_state:
        return

    projects = st.session_state.tailor_projects
    jd_keywords = st.session_state.get("tailor_jd_keywords", [])
    baseline_score = st.session_state.get("tailor_baseline_score", 0)
    st.success(f"✅ Found {len(projects)} project(s)")

    ats_container = st.container(border=True)
    st.divider()

    final_snippets = []

    for idx, proj in enumerate(projects):
        st.subheader(f"📁 Project {idx + 1}")

        if proj["suggestions"]:
            st.markdown("**💡 Suggested points to add — check any that are actually true:**")
            for s_idx, suggestion in enumerate(proj["suggestions"]):
                checked = st.checkbox(suggestion, key=f"suggestion_{idx}_{s_idx}")
                if checked:
                    st.session_state.tailor_approved[idx].add(s_idx)
                else:
                    st.session_state.tailor_approved[idx].discard(s_idx)
        else:
            st.caption("No suggestions generated for this project.")

        if proj.get("has_any_metric"):
            st.caption("📊 One or more suggestions above include an illustrative estimated metric — verify it's roughly accurate before including, or edit it to your real number.")
        else:
            st.caption("📊 No quantitative metric applicable for this project's suggestions.")

        if proj.get("missing_skills"):
            st.markdown("**🚧 Skills this JD wants that aren't on your resume:**")
            st.caption(
                "These are real gaps — not something to fake. Where a plausible, honest tie-in "
                "to your actual work exists, we draft a bullet for it below; check it in only "
                "**at your own risk**, and only if it's genuinely true."
            )
            for sk_idx, item in enumerate(proj["missing_skills"]):
                st.warning(f"**{item['skill']}** — {item['note']}", icon="🚧")
                if "no plausible tie-in" not in item["draft"].lower():
                    checked = st.checkbox(f"💡 {item['draft']}", key=f"skill_{idx}_{sk_idx}")
                    if checked:
                        st.session_state.tailor_approved_skills[idx].add(sk_idx)
                    else:
                        st.session_state.tailor_approved_skills[idx].discard(sk_idx)

        # Nothing changes until the user checks something on - original stays original by default
        approved = st.session_state.tailor_approved[idx]
        approved_skills = st.session_state.tailor_approved_skills[idx]
        approved_bullets = [
            proj["suggestions"][i] for i in sorted(approved) if i < len(proj["suggestions"])
        ] + [
            proj["missing_skills"][i]["draft"] for i in sorted(approved_skills)
            if i < len(proj["missing_skills"])
        ]
        # Insert each approved bullet right next to the existing line it's most related to,
        # rather than always tacking everything onto the end.
        final_snippet = insert_naturally(proj["original"], approved_bullets)
        final_snippets.append(final_snippet)

        st.caption("Side-by-side diff — stays identical until you check a box above, then updates live:")
        left_html, right_html = render_side_by_side_diff(proj["original"], final_snippet)
        col_before, col_after = st.columns(2, gap="medium")
        with col_before:
            st.markdown("**Before**")
            with st.container(border=True):
                st.markdown(left_html, unsafe_allow_html=True)
        with col_after:
            st.markdown("**After**")
            with st.container(border=True):
                st.markdown(right_html, unsafe_allow_html=True)

        st.divider()

    # Current resume text = original with each project's approved edits swapped in - used to
    # score against the JD live, exactly reflecting whatever's checked right now.
    current_text = st.session_state.tailor_full_text
    for idx, proj in enumerate(projects):
        current_text = current_text.replace(proj["original"], final_snippets[idx], 1)
    _ats_score_block(current_text, jd_keywords, baseline_score, ats_container)

    if st.button("👁️ Preview Tailored Resume", use_container_width=True, type="primary"):
        replacements = [
            (proj["original"], final_snippets[idx]) for idx, proj in enumerate(projects)
        ]
        st.session_state.tailor_replacements = replacements

    if "tailor_replacements" in st.session_state:
        replacements = st.session_state.tailor_replacements
        st.markdown("**📄 Preview — this is the structure/formatting the downloaded .docx will have:**")
        with st.container(border=True):
            preview_md = render_tailored_preview_markdown(st.session_state.tailor_full_text, replacements)
            st.markdown(preview_md)

        docx_buffer = build_tailored_docx(st.session_state.tailor_full_text, replacements)
        st.download_button(
            "⬇️ Download Tailored Resume (.docx)",
            data=docx_buffer,
            file_name="tailored_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
