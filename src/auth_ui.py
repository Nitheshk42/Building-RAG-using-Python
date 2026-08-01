import streamlit as st
from src.auth import create_user, verify_user, get_profile


def display_auth():
    """Shows login/signup. Sets st.session_state.auth_user on success."""
    st.title("📚 StudySage")
    st.caption("Interview prep, built from your own resume.")

    tab_login, tab_signup = st.tabs(["🔑 Login", "🆕 Sign Up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if verify_user(username, password):
                    st.session_state.auth_user = username
                    st.session_state.auth_profile = get_profile(username)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if new_password != confirm_password:
                    st.error("❌ Passwords don't match.")
                else:
                    success, message = create_user(new_username, new_email, new_password)
                    if success:
                        st.success(f"✅ {message} Please log in from the Login tab.")
                    else:
                        st.error(f"❌ {message}")
