from datetime import date, datetime, timedelta, timezone
import hashlib
import hmac
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import gspread
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="SmartCafe AI", page_icon="☕", layout="wide")
MONTH_TEMP = {1: 29, 2: 32, 3: 34, 4: 35, 5: 33, 6: 28, 7: 27, 8: 27, 9: 27, 10: 28, 11: 27, 12: 27}
FIXED_HOLIDAYS = {(1, 26): "Republic Day", (8, 15): "Independence Day", (10, 2): "Gandhi Jayanti"}
FEATURE_COLUMNS = [
    "is_holiday", "month", "avg_temp_c", "wfo_headcount",
    *(f"weekday_num_{weekday}" for weekday in range(5)),
]


def setting(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except StreamlitSecretNotFoundError:
        return os.getenv(name, default)


def next_weekday(start: date) -> date:
    target = start + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def authenticate() -> None:
    username_expected = setting("SMARTCAFE_USERNAME")
    password_expected = setting("SMARTCAFE_PASSWORD")
    if not username_expected or not password_expected:
        st.error("Set SMARTCAFE_USERNAME and SMARTCAFE_PASSWORD in Streamlit secrets.")
        st.stop()
    if st.session_state.get("authenticated", False):
        return
    st.title("SmartCafe AI")
    st.caption("Sign in to view live weekday meal planning.")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if hmac.compare_digest(username, username_expected) and hmac.compare_digest(password, password_expected):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect username or password.")
    st.stop()


def cell(row: dict, *names: str):
    values = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        result = values.get(name.casefold())
        if result not in (None, ""):
            return result
    return None


def parse_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def parse_count(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def parse_meal_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        service_date = parse_date(cell(row, "Service date", "service_date (YYYY-MM-DD)", "service_date"))
        source_counts = {
            "veg_prepared": cell(row, "Veg portions prepared", "veg_prepared"),
            "veg_served": cell(row, "Veg portions served", "veg_served"),
            "nonveg_prepared": cell(row, "Non-veg portions prepared", "nonveg_prepared"),
            "nonveg_served": cell(row, "Non-veg portions served", "nonveg_served"),
        }
        if service_date is None or service_date.weekday() >= 5 or any(value is None for value in source_counts.values()):
            continue
        counts = {key: parse_count(value) for key, value in source_counts.items()}
        if any(value is None or value < 0 for value in counts.values()):
            continue
        if counts["veg_served"] > counts["veg_prepared"] or counts["nonveg_served"] > counts["nonveg_prepared"]:
            continue
        result.append({
            "service_date": service_date,
            **counts,
            "meals_served": counts["veg_served"] + counts["nonveg_served"],
            "waste": counts["veg_prepared"] + counts["nonveg_prepared"] - counts["veg_served"] - counts["nonveg_served"],
        })
    return result


def parse_attendance_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        service_date = parse_date(cell(row, "Date", "Service date", "service_date (YYYY-MM-DD)", "service_date"))
        total_value = cell(row, "Total employee headcount", "total_employee_headcount", "total employees")
        total = parse_count(total_value) if total_value is not None else None
        wfo = parse_count(cell(row, "WFO headcount", "wfo_headcount", "employees WFO", "WFO"))
        if service_date is None or service_date.weekday() >= 5 or wfo is None:
            continue
        if wfo < 0 or (total is not None and (total <= 0 or wfo > total)):
            continue
        result.append({
            "service_date": service_date,
            "total_employee_headcount": total,
            "wfo_headcount": wfo,
        })
    return result


@st.cache_resource
def sheets_client():
    account = dict(st.secrets["google_service_account"])
    credentials = Credentials.from_service_account_info(
        account,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    return gspread.authorize(credentials)


@st.cache_data(ttl=30)
def read_live_records() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    sheet_id = setting("VENDOR_RESPONSE_SHEET_ID")
    if not sheet_id:
        raise ValueError("VENDOR_RESPONSE_SHEET_ID is not configured.")
    workbook = sheets_client().open_by_key(sheet_id)
    rows = []
    try:
        rows.extend(workbook.worksheet("Historical Import").get_all_records())
    except gspread.WorksheetNotFound:
        pass
    response_tab = setting("VENDOR_RESPONSE_TAB", "Form Responses 1")
    rows.extend(workbook.worksheet(response_tab).get_all_records())

    attendance_rows = parse_attendance_rows(rows)
    attendance_sheet_id = setting("ATTENDANCE_SHEET_ID")
    attendance_workbook = sheets_client().open_by_key(attendance_sheet_id) if attendance_sheet_id else workbook
    attendance_tab = setting("ATTENDANCE_TAB", "Attendance")
    try:
        attendance_sheet = attendance_workbook.worksheet(attendance_tab)
        attendance_rows.extend(parse_attendance_rows(attendance_sheet.get_all_records()))
    except gspread.WorksheetNotFound:
        pass

    meals = pd.DataFrame(parse_meal_rows(rows))
    attendance = pd.DataFrame(attendance_rows)
    if not meals.empty:
        meals = meals.sort_values("service_date").drop_duplicates("service_date", keep="last").reset_index(drop=True)
    if not attendance.empty:
        attendance = attendance.sort_values("service_date").drop_duplicates("service_date", keep="last").reset_index(drop=True)
    return meals, attendance, datetime.now(timezone.utc).isoformat()


def fetch_json(url: str):
    request = Request(url, headers={"User-Agent": "SmartCafeAI/1.0"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())


@st.cache_data(ttl=86400)
def holidays_for_year(year: int) -> dict[str, str]:
    holidays = {f"{year}-{month:02d}-{day:02d}": name for (month, day), name in FIXED_HOLIDAYS.items()}
    try:
        for item in fetch_json(f"https://date.nager.at/api/v3/PublicHolidays/{year}/IN"):
            if not item.get("counties") or "IN-KA" in item["counties"]:
                holidays[item["date"]] = item.get("localName") or item["name"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return holidays


@st.cache_data(ttl=1800)
def weather_for_date(target_date: str) -> tuple[float, float | None, str]:
    parameters = urlencode({
        "latitude": 12.9716,
        "longitude": 77.5946,
        "daily": "temperature_2m_mean,precipitation_sum",
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "Asia/Kolkata",
    })
    try:
        daily = fetch_json(f"https://api.open-meteo.com/v1/forecast?{parameters}")["daily"]
        temperature, precipitation = daily["temperature_2m_mean"][0], daily["precipitation_sum"][0]
        if temperature is not None:
            return float(temperature), precipitation, "Open-Meteo forecast"
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    return float(MONTH_TEMP[date.fromisoformat(target_date).month]), None, "Bengaluru monthly climatology fallback"


def prepare_training_data(meals: pd.DataFrame, attendance: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if meals.empty:
        raise ValueError("No valid faculty meal records are present in the Form response Sheet.")
    history = meals.merge(attendance, on="service_date", how="left") if not attendance.empty else meals.copy()
    matched = int(history["wfo_headcount"].notna().sum()) if "wfo_headcount" in history else 0
    dates = pd.to_datetime(history["service_date"])
    history["weekday_num"] = dates.dt.weekday
    history["month"] = dates.dt.month
    history["avg_temp_c"] = history["month"].map(MONTH_TEMP).astype(float)
    years = dates.dt.year.unique()
    holiday_dates = {f"{year}-{month:02d}-{day:02d}" for year in years for (month, day) in FIXED_HOLIDAYS}
    for year in years:
        holiday_dates.update(holidays_for_year(int(year)).keys())
    history["is_holiday"] = history["service_date"].map(lambda day: int(day.isoformat() in holiday_dates))
    if "wfo_headcount" not in history:
        history["wfo_headcount"] = np.nan
    history["wfo_headcount"] = history["wfo_headcount"].fillna(history["wfo_headcount"].mean()).fillna(0)
    return history.sort_values("service_date").reset_index(drop=True), matched


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(
        frame[["weekday_num", "month", "avg_temp_c", "is_holiday", "wfo_headcount"]],
        columns=["weekday_num"],
    )
    return encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def train_models(history: pd.DataFrame) -> dict:
    if len(history) < 5:
        raise ValueError(f"At least 5 real weekday meal reports are required; found {len(history)}.")
    features = make_features(history)
    meals_target = history["meals_served"]
    predictions, actual = [], []
    splits = TimeSeriesSplit(n_splits=min(5, len(history) - 1))
    for train_indices, test_indices in splits.split(features):
        validator = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        validator.fit(features.iloc[train_indices], meals_target.iloc[train_indices])
        predictions.extend(validator.predict(features.iloc[test_indices]))
        actual.extend(meals_target.iloc[test_indices])
    models = {
        target: RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42).fit(features, history[target])
        for target in ("meals_served", "veg_prepared", "nonveg_prepared")
    }
    return {
        "models": models,
        "validation_mae": float(mean_absolute_error(actual, predictions)),
        "training_rows": len(history),
        "matched_attendance_rows": matched_rows(history),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def matched_rows(history: pd.DataFrame) -> int:
    return int(history["wfo_headcount"].notna().sum())


def forecast_features(target: date, is_holiday: bool, temperature: float, wfo: float) -> pd.DataFrame:
    return make_features(pd.DataFrame([{
        "weekday_num": target.weekday(), "month": target.month,
        "avg_temp_c": temperature, "is_holiday": int(is_holiday),
        "wfo_headcount": wfo,
    }]))


def frame_signature(meals: pd.DataFrame, attendance: pd.DataFrame) -> str:
    content = meals.to_json(orient="records", date_format="iso") + attendance.to_json(orient="records", date_format="iso")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@st.fragment(run_every="60s")
def live_dashboard(target_date: date, holiday_choice: str) -> None:
    force_refresh = st.session_state.pop("force_sheet_refresh", False)
    force_retrain = st.session_state.pop("force_model_retrain", False)
    if force_refresh or force_retrain:
        read_live_records.clear()
    try:
        meals, attendance, checked_at = read_live_records()
    except (ValueError, KeyError, gspread.GSpreadException, OSError, StreamlitSecretNotFoundError) as error:
        st.error(f"Google Sheets could not be read: {error}")
        st.info("Add the service-account JSON and Sheet ID to Streamlit secrets, then share the response Sheet with the service account email.")
        return

    latest_meal_date = meals["service_date"].max().isoformat() if not meals.empty else "None"
    latest_attendance_date = attendance["service_date"].max().isoformat() if not attendance.empty else "None"
    status = st.columns(4)
    status[0].metric("Valid meal weekdays", len(meals))
    status[1].metric("Attendance weekdays", len(attendance))
    status[2].metric("Latest meal service date", latest_meal_date)
    status[3].metric("Latest attendance service date", latest_attendance_date)
    st.caption(f"Google Sheets checked {checked_at} · polling every 60 seconds while this page is open.")

    if len(meals) < 5:
        st.warning(f"Need at least 5 valid weekday meal reports to train; currently have {len(meals)}.")
        return

    signature = frame_signature(meals, attendance)
    previous_signature = st.session_state.get("model_signature")
    if force_retrain or signature != previous_signature:
        try:
            with st.spinner("Training the model from current real Form and attendance records..."):
                history, matched = prepare_training_data(meals, attendance)
                bundle = train_models(history)
                bundle["matched_attendance_rows"] = matched
                st.session_state["model_bundle"] = bundle
                st.session_state["model_signature"] = signature
            st.success(
                f"Model trained on {bundle['training_rows']} real weekdays; "
                f"attendance matched on {matched}; forward-chaining MAE {bundle['validation_mae']:.1f} meals."
            )
        except ValueError as error:
            st.error(f"Model training failed: {error}")
            return
    elif previous_signature:
        st.caption("No new or edited Sheet records since the last model training.")

    if attendance.empty:
        st.warning("Meal history is available, but no valid WFO attendance records were found; forecasts are paused until attendance is entered.")
        return
    if target_date.weekday() >= 5:
        st.error("The cafeteria operates Monday through Friday. Choose a weekday.")
        return

    holiday_map = holidays_for_year(target_date.year)
    holiday_name = holiday_map.get(target_date.isoformat())
    is_holiday = holiday_name is not None if holiday_choice == "Automatic calendar" else holiday_choice == "Company holiday"
    temperature, precipitation, weather_source = weather_for_date(target_date.isoformat())
    recent = attendance[
        (attendance["service_date"].map(lambda day: day.weekday()) == target_date.weekday())
        & (attendance["service_date"] < target_date)
    ].tail(8)
    selected = recent if not recent.empty else attendance.tail(20)
    wfo_estimate = float(selected["wfo_headcount"].mean())
    feature_row = forecast_features(target_date, is_holiday, temperature, wfo_estimate)
    models = st.session_state["model_bundle"]["models"]
    meals_served = max(0, int(round(models["meals_served"].predict(feature_row)[0])))
    veg_prepared = max(0, int(round(models["veg_prepared"].predict(feature_row)[0])))
    nonveg_prepared = max(0, int(round(models["nonveg_prepared"].predict(feature_row)[0])))
    preparation_total = veg_prepared + nonveg_prepared
    buffer = round(preparation_total * (0.05 if is_holiday else 0.03))

    st.subheader(f"Forecast for {target_date.strftime('%A, %d %B %Y')}")
    metrics = st.columns(5)
    metrics[0].metric("Expected WFO", f"{wfo_estimate:,.0f}")
    metrics[1].metric("Expected meals served", f"{meals_served:,}")
    metrics[2].metric("Veg to prepare", f"{veg_prepared:,}")
    metrics[3].metric("Non-veg to prepare", f"{nonveg_prepared:,}")
    metrics[4].metric("Prep recommendation", f"{preparation_total + buffer:,}", f"{buffer:,} buffer")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recorded weekday trends")
        chart = attendance.merge(meals, on="service_date", how="outer").sort_values("service_date").tail(60)
        chart = chart.set_index("service_date")
        chart_columns = [column for column in ("wfo_headcount", "meals_served") if column in chart]
        if chart_columns:
            st.line_chart(chart[chart_columns], height=280)
    with right:
        bundle = st.session_state["model_bundle"]
        st.subheader("Live model details")
        st.metric("Real meal rows trained", bundle["training_rows"])
        st.metric("Rows matched to attendance", bundle["matched_attendance_rows"])
        st.metric("Forward-chaining MAE", f"{bundle['validation_mae']:.1f} meals")
        st.caption(f"Attendance source: {len(recent)} matching weekday records" if not recent.empty else "Attendance source: recent weekday records")
        st.caption(f"Weather: {weather_source} · {temperature:.1f} °C" + (f" · {precipitation:.1f} mm rain" if precipitation is not None else ""))
        st.caption(f"Calendar: {holiday_name if is_holiday and holiday_name else 'normal workday'}")
        st.caption(f"Last trained: {bundle['trained_at']}")


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
    with st.sidebar:
        if st.button("Sign out"):
            st.session_state.pop("authenticated", None)
            st.session_state.pop("model_bundle", None)
            st.session_state.pop("model_signature", None)
            st.rerun()
        st.header("Forecast")
        target_date = st.date_input("Service date", value=next_weekday(date.today()))
        holiday_choice = st.selectbox("Holiday handling", ["Automatic calendar", "Company holiday", "Normal workday"])
        left, right = st.columns(2)
        if left.button("Refresh data"):
            st.session_state["force_sheet_refresh"] = True
        if right.button("Retrain now", type="primary"):
            st.session_state["force_model_retrain"] = True
        form_url = setting("FACULTY_FORM_URL")
        if form_url:
            st.link_button("Faculty daily form", form_url, use_container_width=True)

    st.title("SmartCafe AI")
    st.caption("Weekday meal planning · live faculty Form and attendance data")
    st.info("Form responses are stored in Google Sheets. This dashboard polls the Sheets while open and retrains when real records change.")
    live_dashboard(target_date, holiday_choice)


if __name__ == "__main__":
    authenticate()
    render_app()
