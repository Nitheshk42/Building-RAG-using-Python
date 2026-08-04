import streamlit as st
from src.resume_tailor import analyze_resume_for_jd, render_side_by_side_diff, build_tailored_docx


def display_resume_tailor():
    st.title("🎯 Resume Tailor")
    st.markdown(
        "Paste a job description — this tailors your **first two projects** to match it. "
        "Rephrasing only uses what's already true in your resume. Anything inferred beyond "
        "that, any missing skill, and any suggested metric is shown separately — you decide "
        "what goes in, and the diff below updates live as you check things on."
    )
    st.divider()

    documents = st.session_state.get("documents")
    if not documents:
        st.warning("⚠️ Please go to Visual tab and process your resume first.")
        return

    full_resume_text = "\n".join(doc.page_content for doc in documents)
    jd_text = st.text_area("Paste the job description here:", height=180, placeholder="Paste JD text...")

    if st.button("🎯 Analyze & Tailor", use_container_width=True):
        if not jd_text.strip():
            st.error("❌ Please paste a job description first.")
            return
        with st.spinner("🔍 Matching your first two projects against this JD..."):
            projects = analyze_resume_for_jd(full_resume_text, jd_text)
        if not projects:
            st.error("❌ Couldn't parse a result — try again.")
            return
        st.session_state.tailor_projects = projects
        st.session_state.tailor_full_text = full_resume_text
        st.session_state.tailor_approved = [set() for _ in projects]
        st.session_state.tailor_approved_metrics = [set() for _ in projects]

    if "tailor_projects" not in st.session_state:
        return

    projects = st.session_state.tailor_projects
    st.success(f"✅ Found {len(projects)} project(s) to tailor")

    final_snippets = []

    for idx, proj in enumerate(projects):
        st.subheader(f"📁 Project {idx + 1}")

        if proj["suggestions"]:
            st.markdown("**⚠️ Suggested additions — verify each is actually true before including:**")
            for s_idx, suggestion in enumerate(proj["suggestions"]):
                checked = st.checkbox(suggestion, key=f"suggestion_{idx}_{s_idx}")
                if checked:
                    st.session_state.tailor_approved[idx].add(s_idx)
                else:
                    st.session_state.tailor_approved[idx].discard(s_idx)

        if proj.get("metric_recommendations"):
            st.markdown("**📊 Quantitative metric recommendations:**")
            st.caption(
                "These are illustrative estimated ranges, not your real numbers — we don't know "
                "your actual results. Include one **at your own risk**, and only if it's roughly "
                "accurate or you replace it with your real figure. Never present an unverified "
                "estimate as fact to a recruiter."
            )
            for m_idx, metric in enumerate(proj["metric_recommendations"]):
                checked = st.checkbox(f"📊 {metric}", key=f"metric_{idx}_{m_idx}")
                if checked:
                    st.session_state.tailor_approved_metrics[idx].add(m_idx)
                else:
                    st.session_state.tailor_approved_metrics[idx].discard(m_idx)

        if proj.get("missing_skills"):
            st.markdown("**🚧 Skills this JD wants that aren't on your resume:**")
            st.caption("Informational only — these are never added to your resume automatically.")
            for skill_note in proj["missing_skills"]:
                st.warning(skill_note, icon="🚧")

        # Build the live final text for this project from base edit + approved checkboxes
        approved = st.session_state.tailor_approved[idx]
        approved_metrics = st.session_state.tailor_approved_metrics[idx]
        final_snippet = proj["edited"]
        approved_text = "\n".join(
            proj["suggestions"][i] for i in sorted(approved) if i < len(proj["suggestions"])
        )
        approved_metric_text = "\n".join(
            proj["metric_recommendations"][i] for i in sorted(approved_metrics)
            if i < len(proj["metric_recommendations"])
        )
        if approved_text:
            final_snippet += "\n" + approved_text
        if approved_metric_text:
            final_snippet += "\n" + approved_metric_text
        final_snippets.append(final_snippet)

        st.caption("Side-by-side diff — red/strikethrough = removed, green = added (updates live as you check boxes above):")
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

    any_metrics_approved = any(st.session_state.tailor_approved_metrics[i] for i in range(len(projects)))
    if any_metrics_approved:
        st.warning(
            "⚠️ You've included estimated metric ranges that aren't your verified real numbers — "
            "make sure they're accurate (or close enough) before using this resume anywhere.",
            icon="⚠️"
        )

    if st.button("📥 Generate Tailored Resume (.docx)", use_container_width=True, type="primary"):
        replacements = [
            (proj["original"], final_snippets[idx]) for idx, proj in enumerate(projects)
        ]
        docx_buffer = build_tailored_docx(st.session_state.tailor_full_text, replacements)
        st.download_button(
            "⬇️ Download Tailored Resume",
            data=docx_buffer,
            file_name="tailored_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
