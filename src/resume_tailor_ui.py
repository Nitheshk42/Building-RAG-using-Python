import streamlit as st
from src.resume_tailor import analyze_resume_for_jd, render_word_diff, build_tailored_docx


def display_resume_tailor():
    st.title("🎯 Resume Tailor")
    st.markdown(
        "Paste a job description — this tailors your **first two projects** to match it. "
        "Rephrasing only uses what's already true in your resume. Anything inferred beyond "
        "that is shown separately as a **suggestion you must verify and approve** before it "
        "goes into the download — nothing gets added silently."
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

    if "tailor_projects" not in st.session_state:
        return

    projects = st.session_state.tailor_projects
    st.success(f"✅ Found {len(projects)} project(s) to tailor")

    for idx, proj in enumerate(projects):
        st.subheader(f"📁 Project {idx + 1}")

        st.caption("Diff view — red/strikethrough = removed, green = added (only rewording of what's already true):")
        with st.container(border=True):
            st.markdown(render_word_diff(proj["original"], proj["edited"]), unsafe_allow_html=True)

        with st.expander("📄 Before / After (plain text)"):
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**Before:**")
                st.text(proj["original"])
            with col_after:
                st.markdown("**After:**")
                st.text(proj["edited"])

        if proj["suggestions"]:
            st.markdown("**⚠️ Suggested additions — verify each is actually true before including:**")
            for s_idx, suggestion in enumerate(proj["suggestions"]):
                checked = st.checkbox(suggestion, key=f"suggestion_{idx}_{s_idx}")
                if checked:
                    st.session_state.tailor_approved[idx].add(s_idx)
                else:
                    st.session_state.tailor_approved[idx].discard(s_idx)

        st.divider()

    if st.button("📥 Generate Tailored Resume (.docx)", use_container_width=True, type="primary"):
        replacements = []
        for idx, proj in enumerate(projects):
            approved = st.session_state.tailor_approved[idx]
            approved_text = "\n".join(
                proj["suggestions"][i] for i in sorted(approved) if i < len(proj["suggestions"])
            )
            final_snippet = proj["edited"]
            if approved_text:
                final_snippet += "\n" + approved_text
            replacements.append((proj["original"], final_snippet))

        docx_buffer = build_tailored_docx(st.session_state.tailor_full_text, replacements)
        st.download_button(
            "⬇️ Download Tailored Resume",
            data=docx_buffer,
            file_name="tailored_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
