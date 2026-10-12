from datetime import date, timedelta
import hmac
import os

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

def setting(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except StreamlitSecretNotFoundError:
        return os.getenv(name, default)


def next_weekday(start_date: date) -> date:
    target = start_date + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def authenticate() -> None:
    expected_username = setting("SMARTCAFE_USERNAME")
    expected_password = setting("SMARTCAFE_PASSWORD")
    if not expected_username or not expected_password:
        st.error("App access is not configured. Add SMARTCAFE_USERNAME and SMARTCAFE_PASSWORD to deployment secrets.")
        st.stop()
    if st.session_state.get("authenticated", False):
        return

    st.title("SmartCafe AI")
    st.caption("Sign in to access live attendance and meal forecasts.")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        username_matches = hmac.compare_digest(username, expected_username)
        password_matches = hmac.compare_digest(password, expected_password)
        if username_matches and password_matches:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect username or password.")
    st.stop()


def api_get(api_url: str, api_key: str, path: str) -> dict:
    response = requests.get(
        api_url + path,
        headers={"X-API-Key": api_key},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def api_post(api_url: str, api_key: str, path: str, payload: dict) -> dict:
    response = requests.post(
        api_url + path,
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def render_app() -> None:
    st.markdown("""
    <style>
        .stApp { background: #f4f6f2; }
        [data-testid="stHeader"] { background: rgba(244, 246, 242, 0.92); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #173b32; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid #dce5dc; padding: 1rem; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    api_url = setting("SMARTCAFE_API_URL").rstrip("/")
    api_key = setting("SMARTCAFE_API_KEY")
    form_url = setting("FACULTY_FORM_URL")
    if not api_url or not api_key:
        st.error("API is not configured. Add SMARTCAFE_API_URL and SMARTCAFE_API_KEY to deployment secrets.")
        st.stop()

    with st.sidebar:
        if st.button("Sign out"):
            st.session_state.pop("authenticated", None)
            st.rerun()
        st.header("Forecast")
        target_date = st.date_input("Service date", value=next_weekday(date.today()))
        holiday_choice = st.selectbox(
            "Holiday handling",
            ["Automatic calendar", "Company holiday", "Normal workday"],
        )
        refresh = st.button("Refresh live data", use_container_width=True)
        retrain = st.button("Retrain model now", type="primary", use_container_width=True)
        if form_url:
            st.link_button("Open faculty daily form", form_url, use_container_width=True)

    if target_date.weekday() >= 5:
        st.error("The cafeteria operates Monday through Friday. Choose a weekday.")
        st.stop()

    try:
        if retrain:
            with st.spinner("Retraining from saved faculty and attendance records..."):
                result = api_post(api_url, api_key, "/retrain", {})
            st.success(
                f"Retrained from {result['real_training_rows']} real meal rows; "
                f"attendance matched on {result['rows_with_matched_attendance']}; "
                f"forward-chaining MAE {result['validation_mae_meals_served']:.1f} meals."
            )
        service_status = api_get(api_url, api_key, "/")
        live_status = api_get(api_url, api_key, "/status")
    except requests.RequestException as error:
        st.error(f"The live model API is unavailable at {api_url}: {error}")
        st.stop()

    st.title("SmartCafe AI")
    st.caption("Weekday meal planning · faculty-submitted live records")
    st.info("Every faculty Form submission is sent to the API immediately and triggers automatic model retraining.")

    status_cols = st.columns(4)
    status_cols[0].metric("Faculty meal reports", live_status["vendor_rows"])
    status_cols[1].metric("Attendance records", live_status["attendance_rows"])
    status_cols[2].metric("Real training rows", live_status["real_training_rows"])
    status_cols[3].metric("Rows matched to attendance", live_status["matched_attendance_rows"])
    st.caption(
        "Last report received: " + str(live_status["vendor_last_received_at"] or "none")
        + " · last attendance received: " + str(live_status["attendance_last_received_at"] or "none")
        + " · model updated: " + str(live_status["model_updated_at"] or "not recorded")
    )

    if live_status["vendor_rows"] == 0 or live_status["attendance_rows"] == 0:
        st.warning("No live faculty records have reached this API yet. Submit the Form or import weekday history before using forecasts.")
        if refresh:
            st.toast("Status refreshed from the API.")
        st.stop()

    override = None
    if holiday_choice == "Company holiday":
        override = True
    elif holiday_choice == "Normal workday":
        override = False
    payload = {"target_date": target_date.isoformat()}
    if override is not None:
        payload["is_holiday"] = override
    try:
        prediction = api_post(api_url, api_key, "/predict", payload)
    except requests.RequestException as error:
        st.error(f"Forecast request failed: {error}")
        st.stop()

    st.subheader(f"Forecast for {prediction['weekday']}, {target_date:%d %B %Y}")
    metrics = st.columns(5)
    metrics[0].metric("Expected WFO", f"{prediction['expected_attendance']:,}")
    metrics[1].metric("Expected meals served", f"{prediction['expected_meals_served']:,}")
    metrics[2].metric("Veg to prepare", f"{prediction['expected_veg']:,}")
    metrics[3].metric("Non-veg to prepare", f"{prediction['expected_nonveg']:,}")
    metrics[4].metric(
        "Prep recommendation",
        f"{prediction['prepared_recommendation']:,}",
        f"{prediction['buffer']:,} buffer",
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recorded weekday trends")
        try:
            records = api_get(api_url, api_key, "/history").get("records", [])
            if records:
                history = pd.DataFrame(records)
                history["service_date"] = pd.to_datetime(history["service_date"])
                history = history.set_index("service_date")
                chart_columns = [column for column in ("wfo_headcount", "meals_served") if column in history]
                if chart_columns:
                    st.line_chart(history[chart_columns].tail(60), height=280)
            else:
                st.caption("No faculty Form records have reached the API yet.")
        except requests.RequestException as error:
            st.warning(f"History is unavailable: {error}")
    with right:
        st.subheader("Live model details")
        st.metric("Forward-chaining MAE", f"{prediction['validation_mae_meals_served']:.1f} meals")
        st.caption(f"Weather: {prediction['weather_source']} · {prediction['avg_temp_c']:.1f} °C")
        st.caption(f"Calendar: {prediction['holiday_name'] or 'normal workday'}")
        st.caption(service_status.get("model_quality", {}).get("note", "Validation is based on recorded history."))
    if refresh:
        st.toast("Live status and forecast refreshed from the API.")


if __name__ == "__main__":
    from streamlit_sheets_app import authenticate as sheets_authenticate
    from streamlit_sheets_app import render_app as render_sheets_app

    sheets_authenticate()
    render_sheets_app()
