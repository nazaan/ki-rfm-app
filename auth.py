"""
auth.py
-------
Authentication gate for the RFM app.

CURRENT STATE: Stub — always grants access.
PRODUCTION STATE: Swap _check_subscription() to call the FastAPI
/user/status endpoint. Everything else stays the same.
"""

import streamlit as st
from typing import Optional
import os

STUB_MODE    = True       # set False when platform skeleton is live
STUB_TIER    = "agency"   # gives all features during dev
API_BASE_URL = os.getenv("KI_API_URL", "http://localhost:8000")


def require_subscription(min_tier: str = "small_biz") -> dict:
    if STUB_MODE:
        return _stub_user()
    if "auth_token" in st.session_state and "auth_user" in st.session_state:
        user = st.session_state["auth_user"]
        if _tier_sufficient(user["tier"], min_tier):
            return user
        else:
            _show_upgrade_wall(user["tier"], min_tier)
            st.stop()
    _show_login_ui(min_tier)
    st.stop()


TIER_RANK = {"small_biz": 1, "agency": 2}

def _tier_sufficient(user_tier: str, required_tier: str) -> bool:
    return TIER_RANK.get(user_tier, 0) >= TIER_RANK.get(required_tier, 0)

def is_agency(user: dict) -> bool:
    return user.get("tier") == "agency"

def is_small_biz_or_above(user: dict) -> bool:
    return _tier_sufficient(user.get("tier", ""), "small_biz")


def _stub_user() -> dict:
    user = {"email": "dev@kidatalab.com", "tier": STUB_TIER, "stub": True}
    st.session_state["auth_user"] = user
    return user


def _show_login_ui(min_tier: str):
    st.markdown("## KI DataLab")
    st.markdown("### Sign in to continue")
    st.caption("Enter your email. We'll send you a sign-in link — no password needed.")
    email = st.text_input("Email address", placeholder="you@example.com", key="login_email")
    if st.button("Send sign-in link", type="primary"):
        if email and "@" in email:
            success = _request_magic_link(email)
            if success:
                st.success(f"Check your inbox at **{email}** — link expires in 15 minutes.")
            else:
                st.error("No active subscription found for this email.")
        else:
            st.warning("Please enter a valid email address.")
    st.divider()
    st.caption("Don't have a subscription? [View plans →](https://kidatalab.com/pricing)")


def _show_upgrade_wall(current_tier: str, required_tier: str):
    tier_labels = {"small_biz": "Small Business", "agency": "Agency"}
    st.warning(f"This feature requires the **{tier_labels.get(required_tier)}** plan.")
    st.markdown("[Upgrade your plan →](https://kidatalab.com/pricing)")


def _request_magic_link(email: str) -> bool:
    try:
        import requests
        resp = requests.post(f"{API_BASE_URL}/auth/request", json={"email": email}, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _verify_token(token: str) -> Optional[dict]:
    try:
        import requests
        resp = requests.get(f"{API_BASE_URL}/auth/verify", params={"token": token}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {"email": data["email"], "tier": data["tier"], "stub": False}
        return None
    except Exception:
        return None


def handle_token_from_url():
    if STUB_MODE:
        return
    params = st.query_params
    if "token" in params and "auth_token" not in st.session_state:
        token = params["token"]
        user  = _verify_token(token)
        if user:
            st.session_state["auth_token"] = token
            st.session_state["auth_user"]  = user
            st.query_params.clear()
            st.rerun()
        else:
            st.error("This sign-in link has expired. Please request a new one.")


def logout():
    for key in ["auth_token", "auth_user"]:
        st.session_state.pop(key, None)
    st.rerun()


def show_auth_status_sidebar(user: dict):
    tier_colors  = {"small_biz": "#009084", "agency": "#561269"}
    tier_labels  = {"small_biz": "Small Biz", "agency": "Agency"}
    tier  = user.get("tier", "")
    color = tier_colors.get(tier, "#6B7E75")
    label = tier_labels.get(tier, tier)
    stub  = user.get("stub", False)
    st.sidebar.markdown("---")
    if stub:
        st.sidebar.caption("🔧 Dev mode — auth stub active")
    else:
        st.sidebar.caption(f"Signed in as **{user.get('email', '')}**")
    st.sidebar.markdown(
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:bold">{label}</span>',
        unsafe_allow_html=True,
    )
    if not stub:
        if st.sidebar.button("Sign out", key="signout_btn"):
            logout()
