import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import gradio as gr
import joblib
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_NUMBER = "12301"
MODEL_PATH = "railway_eta_recovered_model.joblib"

RAILRADAR_API_KEY = os.environ.get("RAILRADAR_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

xgb_model = None
preprocessor = None
feature_columns = None


def load_model():
    global xgb_model, preprocessor, feature_columns

    if xgb_model is None:
        print("Loading ETA model...")

        artifact = joblib.load(MODEL_PATH)

        xgb_model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        feature_columns = artifact["feature_columns"]

        print("ETA model loaded successfully.")


# ============================================================
# HISTORICAL PROFILE
# Derived from 13 historical 12301 journeys
# with distance between 1400 and 1500 km.
# ============================================================

HISTORICAL_PROFILE = {
    "scheduled_travel_hours": 20.01,
    "track_doubled": 1,
    "is_hdn_route": 0,
    "traction_type": "Electric (25kV AC)",
    "is_electrified": 1,
    "psr_count": 3,
    "is_circular_route": 0,
    "zone_fog_index": 0.22,
    "zone_congestion_index": 0.74,
    "season_severity_score": 0.72,
    "loco_age_years": 11.5,
    "coach_age_years": 14.8,
    "has_lhb_coaches": 1,
    "is_rake_shared": 0,
    "maintenance_score": 6.8,
    "seat_utilisation_pct": 81.0,
    "is_overloaded": 0,
    "is_special_train": 0,
    "route_historical_ontime_pct": 74.1,
}


# ============================================================
# HELPER: SEASON
# ============================================================

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter/Fog"
    elif month in [3, 4]:
        return "Pre-Monsoon"
    elif month in [5, 6]:
        return "Summer"
    elif month in [7, 8]:
        return "Monsoon"
    elif month == 9:
        return "Post-Monsoon"
    else:
        return "Autumn"


# ============================================================
# HELPER: WEATHER -> FOG RISK
# ============================================================

def calculate_fog_risk(visibility_km, humidity):
    if visibility_km >= 5:
        return 0.0

    visibility_component = max(
        0,
        min(1, (5 - visibility_km) / 5)
    )

    humidity_component = max(
        0,
        min(1, (humidity - 70) / 30)
    )

    raw_score = (
        0.6 * visibility_component
        + 0.4 * humidity_component
    )

    score = 0.32 + raw_score * (0.88 - 0.32)

    return float(max(0.32, min(0.88, score)))


# ============================================================
# LIVE SEGMENT HELPERS
# ============================================================


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt

    try:
        lat1, lon1, lat2, lon2 = map(
            float, (lat1, lon1, lat2, lon2)
        )
    except (TypeError, ValueError):
        return None

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    return 2 * r * asin(sqrt(a))


def get_segment_context(current, route, lat, lon):
    """Return a human-readable current segment and its approximate length."""
    current_code = current.get("stationCode")
    current_seq = current.get("sequence")

    current_index = None

    for i, stop in enumerate(route):
        if current_code and stop.get("stationCode") == current_code:
            current_index = i
            break
        if current_seq is not None and stop.get("sequence") == current_seq:
            current_index = i
            break

    next_stop = None
    previous_stop = None

    if current_index is not None:
        if current_index + 1 < len(route):
            next_stop = route[current_index + 1]
        if current_index > 0:
            previous_stop = route[current_index - 1]

    # Some RailRadar responses may explicitly provide next station info.
    next_code = (
        current.get("nextStationCode")
        or current.get("nextStation", {}).get("stationCode")
        if isinstance(current.get("nextStation"), dict)
        else current.get("nextStationCode")
    )

    if next_code and current_index is not None:
        for stop in route:
            if stop.get("stationCode") == next_code:
                next_stop = stop
                break

    start_name = (
        previous_stop.get("stationName")
        if previous_stop
        else current.get("stationName", "Current location")
    )
    end_name = (
        next_stop.get("stationName")
        if next_stop
        else "Next station"
    )

    segment_label = f"{start_name} → {end_name}"

    segment_distance_km = None

    if previous_stop and next_stop:
        d1 = _safe_float(previous_stop.get("distance"))
        d2 = _safe_float(next_stop.get("distance"))
        if d1 is not None and d2 is not None:
            segment_distance_km = abs(d2 - d1)

    if segment_distance_km is None and next_stop:
        segment_distance_km = _haversine_km(
            lat,
            lon,
            next_stop.get("lat"),
            next_stop.get("lng"),
        )

    if segment_distance_km is None or segment_distance_km <= 0:
        segment_distance_km = None

    return segment_label, segment_distance_km


# ============================================================
# MAIN LIVE PREDICTION
# ============================================================

def get_live_prediction():
    load_model()

    # --------------------------------------------------------
    # Check API keys
    # --------------------------------------------------------

    if not RAILRADAR_API_KEY:
        return (
            "## ❌ RailRadar API key missing\n\n"
            "Add `RAILRADAR_API_KEY` to the Render Environment Variables."
        )

    if not OPENWEATHER_API_KEY:
        return (
            "## ❌ OpenWeather API key missing\n\n"
            "Add `OPENWEATHER_API_KEY` to the Render Environment Variables."
        )

    # ========================================================
    # 1. RAILRADAR
    # ========================================================

    rail_url = (
        f"https://api.railradar.in/v1/trains/"
        f"{TRAIN_NUMBER}/live?authoritative=true&includeCoordinates=true"
    )

    try:
        rail_response = requests.get(
            rail_url,
            headers={
                "Authorization": f"Bearer {RAILRADAR_API_KEY}"
            },
            timeout=20,
        )
    except Exception as e:
        return (
            "## ❌ RailRadar connection error\n\n"
            f"`{str(e)}`"
        )

    if rail_response.status_code != 200:
        return (
            "## ❌ RailRadar API error\n\n"
            f"**HTTP status:** `{rail_response.status_code}`\n\n"
            f"```text\n{rail_response.text[:500]}\n```"
        )

    try:
        rail_json = rail_response.json()
    except Exception:
        return "## ❌ RailRadar returned invalid JSON."

    if not rail_json.get("success"):
        return (
            "## ❌ RailRadar returned an unsuccessful response.\n\n"
            f"```text\n{rail_json}\n```"
        )

    rail = rail_json["data"]
    current = rail["currentLocation"]
    train_info = rail["train"]

    # ========================================================
    # 2. CURRENT LOCATION
    # ========================================================

    lat = current.get("lat")
    lon = current.get("lng")

    route = rail.get("route", [])
    current_code = current.get("stationCode")

    # Try to find current station coordinates from route.
    if lat is None or lon is None:
        matching = [
            station
            for station in route
            if station.get("stationCode") == current_code
        ]

        if matching:
            lat = matching[0].get("lat")
            lon = matching[0].get("lng")

    # Final fallback to source station.
    if lat is None or lon is None:
        lat = train_info["source"]["lat"]
        lon = train_info["source"]["lng"]

    current_delay = float(
        rail.get("delayMinutes", 0) or 0
    )

    # --------------------------------------------------------
    # LIVE SEGMENT PROGRESS
    # --------------------------------------------------------

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    raw_segment_progress = current.get("segmentProgress")
    segment_progress = _safe_float(raw_segment_progress)

    if segment_progress is not None:
        if segment_progress > 1:
            segment_progress = segment_progress / 100.0
        segment_progress = max(0.0, min(1.0, segment_progress))
        segment_progress_display = f"{segment_progress * 100:.0f}%"
    else:
        segment_progress_display = "Unavailable"

    segment_label, segment_distance_km = get_segment_context(
        current,
        route,
        lat,
        lon,
    )

    # ========================================================
    # 3. OPENWEATHER
    # ========================================================

    weather_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    try:
        weather_response = requests.get(
            weather_url,
            timeout=20,
        )
    except Exception as e:
        return (
            "## ❌ OpenWeather connection error\n\n"
            f"`{str(e)}`"
        )

    if weather_response.status_code != 200:
        return (
            "## ❌ OpenWeather API error\n\n"
            f"**HTTP status:** `{weather_response.status_code}`\n\n"
            f"```text\n{weather_response.text[:500]}\n```"
        )

    try:
        weather = weather_response.json()
    except Exception:
        return "## ❌ OpenWeather returned invalid JSON."

    weather_main = weather["main"]
    weather_info = weather["weather"][0]
    weather_wind = weather.get("wind", {})

    temperature = float(weather_main.get("temp", 0))
    humidity = float(weather_main.get("humidity", 0))
    pressure = float(weather_main.get("pressure", 0))

    visibility_km = (
        weather.get("visibility", 10000) / 1000
    )

    wind_speed = float(weather_wind.get("speed", 0))

    weather_condition = weather_info.get(
        "description",
        "unknown",
    )

    # ========================================================
    # 4. CURRENT DATE/TIME
    # ========================================================

    year = now.year
    month = now.month
    day_of_week = now.weekday()
    departure_hour = now.hour

    is_weekend = int(day_of_week >= 5)

    is_night_departure = int(
        departure_hour < 6 or departure_hour >= 22
    )

    is_peak_hour = int(
        departure_hour in [7, 8, 9, 17, 18, 19, 20]
    )

    is_festival_season = int(
        month in [10, 11]
    )

    season = get_season(month)

    is_monsoon_season = int(
        month in [7, 8]
    )

    # ========================================================
    # 5. WEATHER FEATURE
    # ========================================================

    fog_risk_score = calculate_fog_risk(
        visibility_km,
        humidity,
    )

    # ========================================================
    # 6. BUILD 36 MODEL FEATURES
    # ========================================================

    features = {
        # Categorical
        "train_number": 12301,
        "train_type": "Rajdhani Express",
        "season": season,
        "traction_type": HISTORICAL_PROFILE["traction_type"],
        "primary_delay_cause": "On Time",

        # Time
        "year": year,
        "month": month,
        "day_of_week": day_of_week,
        "departure_hour": departure_hour,
        "is_weekend": is_weekend,
        "is_night_departure": is_night_departure,
        "is_peak_hour": is_peak_hour,
        "is_festival_season": is_festival_season,

        # Route
        "distance_km": float(
            train_info.get("distance", 1450.1)
        ),
        "num_scheduled_stops": int(
            train_info.get("totalHalts", 9)
        ),
        "scheduled_travel_hours": HISTORICAL_PROFILE[
            "scheduled_travel_hours"
        ],

        # Infrastructure
        "track_doubled": HISTORICAL_PROFILE["track_doubled"],
        "is_hdn_route": HISTORICAL_PROFILE["is_hdn_route"],
        "is_electrified": HISTORICAL_PROFILE["is_electrified"],
        "psr_count": HISTORICAL_PROFILE["psr_count"],
        "is_circular_route": HISTORICAL_PROFILE["is_circular_route"],

        # Weather / season
        "is_monsoon_season": is_monsoon_season,
        "fog_risk_score": fog_risk_score,
        "zone_fog_index": HISTORICAL_PROFILE["zone_fog_index"],
        "zone_congestion_index": HISTORICAL_PROFILE[
            "zone_congestion_index"
        ],
        "season_severity_score": HISTORICAL_PROFILE[
            "season_severity_score"
        ],

        # Train condition
        "loco_age_years": HISTORICAL_PROFILE["loco_age_years"],
        "coach_age_years": HISTORICAL_PROFILE["coach_age_years"],
        "has_lhb_coaches": HISTORICAL_PROFILE["has_lhb_coaches"],
        "is_rake_shared": HISTORICAL_PROFILE["is_rake_shared"],
        "maintenance_score": HISTORICAL_PROFILE["maintenance_score"],
        "seat_utilisation_pct": HISTORICAL_PROFILE[
            "seat_utilisation_pct"
        ],
        "is_overloaded": HISTORICAL_PROFILE["is_overloaded"],
        "is_special_train": HISTORICAL_PROFILE["is_special_train"],

        # Historical performance
        "route_historical_ontime_pct": HISTORICAL_PROFILE[
            "route_historical_ontime_pct"
        ],

        # Live delay state
        "is_delayed": int(current_delay > 0),
    }

    # ========================================================
    # 7. MODEL INPUT
    # ========================================================

    data = pd.DataFrame([features])
    data = data[feature_columns]

    # ========================================================
    # 8. PREPROCESS + XGBOOST
    # ========================================================

    X = preprocessor.transform(data)

    predicted_delay = float(
        xgb_model.predict(X)[0]
    )

    predicted_delay = max(
        0.0,
        predicted_delay,
    )

    # --------------------------------------------------------
    # Model-attributed fog effect (counterfactual)
    # --------------------------------------------------------
    # This is not a causal guarantee. It measures how the trained
    # model's prediction changes when the live fog-risk feature is
    # replaced with a zero-fog scenario while other inputs remain fixed.
    fog_counterfactual = data.copy()
    fog_counterfactual["fog_risk_score"] = 0.0

    X_no_fog = preprocessor.transform(fog_counterfactual)
    predicted_delay_no_fog = float(
        xgb_model.predict(X_no_fog)[0]
    )
    predicted_delay_no_fog = max(0.0, predicted_delay_no_fog)

    fog_delay_effect = predicted_delay - predicted_delay_no_fog

    # ========================================================
    # 9. DESTINATION + SCHEDULE
    # ========================================================

    destination = train_info["destination"]
    destination_code = destination["code"]

    destination_stop = None

    for stop in route:
        if stop.get("stationCode") == destination_code:
            destination_stop = stop
            break

    scheduled_arrival = None
    board_expected_time = "—"

    if destination_stop:
        scheduled_arrival = (
            destination_stop.get("scheduledArrival")
            or destination_stop.get("arrivalTime")
        )

    if scheduled_arrival:
        try:
            scheduled_arrival_dt = datetime.fromisoformat(
                scheduled_arrival.replace("Z", "+00:00")
            )

            scheduled_arrival_dt = scheduled_arrival_dt.astimezone(
                ZoneInfo("Asia/Kolkata")
            )

            predicted_eta_dt = (
                scheduled_arrival_dt
                + timedelta(minutes=predicted_delay)
            )

            scheduled_arrival_display = (
                scheduled_arrival_dt.strftime(
                    "%d %b %Y, %I:%M %p"
                )
            )

            predicted_eta_display = (
                predicted_eta_dt.strftime(
                    "%d %b %Y, %I:%M %p"
                )
            )

            board_expected_time = predicted_eta_dt.strftime("%I:%M")

        except Exception:
            scheduled_arrival_display = str(
                scheduled_arrival
            )
            predicted_eta_display = "Unable to calculate"
    else:
        scheduled_arrival_display = "Unavailable"
        predicted_eta_display = "Unavailable"

    # ========================================================
    # 10. USER-FACING RESULT
    # ========================================================

    current_station = current.get(
        "stationName",
        "Unknown",
    )

    platform_value = (
        current.get("platform")
        or current.get("platformNumber")
        or current.get("platformNo")
    )
    platform_display = str(platform_value) if platform_value else "—"

    # Live train status used by the dashboard header and status card.
    status = rail.get("status", "unknown")
    live_status = (
        "LIVE"
        if str(status).lower() in {
            "running",
            "enroute",
            "on-route",
            "active",
        }
        else str(status).title()
    )

    fog_effect_display = (
        f"+{fog_delay_effect:.2f} min"
        if fog_delay_effect >= 0
        else f"{fog_delay_effect:.2f} min"
    )

    result = f"""
<div class="dashboard">

<div class="hero">
    <div class="live-pill">● {live_status.upper()}</div>
    <div class="hero-title">🚆 Smart Rail ETA</div>
    <div class="hero-subtitle">
        AI-powered real-time arrival prediction
    </div>
    <div class="train-route">
        Train <b>12301</b> · Howrah → New Delhi Rajdhani Express
    </div>
</div>

<div class="section-title">📡 Live Train Intelligence</div>

<div class="cards">

<div class="card">
    <div class="card-label">CURRENT LOCATION</div>
    <div class="card-value">{current_station}</div>
</div>

<div class="card">
    <div class="card-label">TRAIN STATUS</div>
    <div class="card-value">{str(status).title()}</div>
</div>

<div class="card">
    <div class="card-label">CURRENT DELAY</div>
    <div class="card-value">{current_delay:.0f} min</div>
</div>

</div>

<div class="cards secondary-cards">

<div class="card small-card">
    <div class="card-label">CURRENT SEGMENT PROGRESS</div>
    <div class="card-value">{segment_progress_display}</div>
    <div class="card-note">{segment_label}</div>
    <div class="card-note">Percentage completed within this current RailRadar route segment.</div>
</div>

<div class="card small-card">
    <div class="card-label">LAST REFRESHED</div>
    <div class="card-value">{now.strftime("%I:%M:%S %p")}</div>
</div>

</div>

<div class="info-panel">
    <span>📍 <b>Live Coordinates</b></span>
    <span>{lat:.5f}, {lon:.5f}</span>
</div>

<div class="section-title">🌦️ Live Weather Intelligence</div>

<div class="cards">

<div class="card">
    <div class="card-label">CONDITION</div>
    <div class="card-value">{weather_condition.title()}</div>
</div>

<div class="card">
    <div class="card-label">TEMPERATURE</div>
    <div class="card-value">{temperature:.1f} °C</div>
</div>

<div class="card">
    <div class="card-label">HUMIDITY</div>
    <div class="card-value">{humidity:.0f}%</div>
</div>

<div class="card">
    <div class="card-label">VISIBILITY</div>
    <div class="card-value">{visibility_km:.1f} km</div>
</div>

<div class="card">
    <div class="card-label">PRESSURE</div>
    <div class="card-value">{pressure:.0f} hPa</div>
</div>

<div class="card">
    <div class="card-label">WIND</div>
    <div class="card-value">{wind_speed:.2f} m/s</div>
</div>

</div>

<div class="section-title">🌫️ Fog Risk Intelligence</div>

<div class="info-panel">
    <span>🌫️ <b>Fog Risk Score</b></span>
    <span class="score">{fog_risk_score:.2f}</span>
</div>

<details class="explain-box fog-box">
<summary>ⓘ How is Fog Risk calculated and how does it affect the AI prediction?</summary>
<div class="explain-content">

<b>Step 1 — Calculate the weather-derived fog feature</b>
<p>If visibility is <b>5 km or more</b>, the fog-risk feature is set to <b>0</b>. Below 5 km, visibility and humidity are combined:</p>
<code>Visibility Component = clip((5 − visibility) / 5, 0, 1)</code><br>
<code>Humidity Component = clip((humidity − 70) / 30, 0, 1)</code><br>
<code>Raw Risk = 0.6 × Visibility Component + 0.4 × Humidity Component</code><br>
<code>Fog Risk = 0.32 + Raw Risk × (0.88 − 0.32)</code>
<p>The final engineered feature is bounded between <b>0.32 and 0.88</b> when the low-visibility calculation is active.</p>

<b>Step 2 — How fog affects the AI prediction</b>
<p>Fog is <b>not manually converted into a fixed number of delay minutes</b>. The fog-risk score is one of the features supplied to the trained XGBoost model.</p>
<code>Model Fog Effect ≈ Prediction(actual fog) − Prediction(fog risk = 0)</code>
<p><b>Current model-attributed effect: {fog_effect_display}</b></p>
<p class="small-explanation">This is a model sensitivity/counterfactual comparison. It shows how the model's output changes when only the fog-risk feature is changed; it is not a causal guarantee that fog alone creates exactly this many minutes of delay.</p>

</div>
</details>

<div class="section-title">🤖 AI Prediction</div>

<div class="prediction-card">
    <div class="prediction-label">PREDICTED FINAL DELAY</div>
    <div class="prediction-value">{predicted_delay:.2f} min</div>
    <div class="prediction-note">
        Predicted final delay from the trained XGBoost model.
    </div>
</div>

<details class="explain-box">
<summary>ⓘ How is the XGBoost prediction calculated?</summary>
<div class="explain-content">

<b>What the model predicts</b>
<p>The trained model predicts <b>final delay in minutes</b>. It does not directly predict the clock time.</p>

<b>Step 1 — Build the 36-feature input</b>
<p>The app combines live train state, live weather-derived information, route characteristics, infrastructure, train condition, historical performance and calendar features.</p>

<b>Step 2 — Preprocess the features</b>
<p>Categorical values are transformed using the <b>saved preprocessing pipeline</b> (one-hot encoding), while numerical features are passed through. The transformed feature vector is then supplied to XGBoost.</p>

<b>Step 3 — XGBoost ensemble</b>
<code>ŷ = base_score + Σ [learning_rate × tree_m(x)]</code>
<p>Conceptually, XGBoost adds the contributions of many decision trees. This trained model uses <b>500 trees</b>, a <b>0.05 learning rate</b> and <b>maximum tree depth 6</b>.</p>

<p>There is <b>no single hand-written equation</b> such as “delay = weather + current delay”. The final number comes from the learned decision-tree ensemble after all 36 features have been processed.</p>

<b>Final ETA</b>
<code>Predicted ETA = Scheduled Arrival + Predicted Final Delay</code>
<p>Example: if scheduled arrival is 10:05 AM and predicted final delay is 21 minutes, the displayed AI ETA is approximately 10:26 AM.</p>

</div>
</details>

<div class="section-title">🕐 Estimated Arrival</div>

<div class="eta-panel">

<div>
    <div class="eta-label">SCHEDULED ARRIVAL</div>
    <div class="eta-time">{scheduled_arrival_display}</div>
</div>

<div class="arrow">→</div>

<div>
    <div class="eta-label">AI PREDICTED ETA</div>
    <div class="eta-time highlight">{predicted_eta_display}</div>
</div>

</div>

<div class="section-title">📟 Railway Display Board — Demo</div>

<div class="railway-board">
    <div class="board-head">
        <span>TRAIN NO.</span>
        <span>TRAIN NAME</span>
        <span>EXPT. TIME</span>
        <span>A/D</span>
        <span>PF. NO.</span>
    </div>
    <div class="board-row">
        <span>12301</span>
        <span>RAJDHANI EXPRESS</span>
        <span>{board_expected_time}</span>
        <span>A</span>
        <span>{platform_display}</span>
    </div>
    <div class="board-note">
        Demo railway display · Expected time uses the AI-predicted ETA · Platform number is shown when the live API provides it; otherwise it is displayed as —.
    </div>
</div>


<div class="footer">
    Last refreshed: {now.strftime("%d %b %Y, %I:%M:%S %p")} IST<br>
    Prediction combines the trained XGBoost model with live RailRadar
    and OpenWeather data.
</div>

</div>
"""

    return result


# ============================================================
# PROFESSIONAL WEB INTERFACE
# ============================================================

CSS = """
body {
    background: #0b1020 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: #0b1020 !important;
}

#refresh-btn {
    width: 100%;
    min-height: 58px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 14px;
    margin-bottom: 18px;
}

.dashboard {
    font-family: Arial, sans-serif;
    color: #172033 !important;
}

.dashboard * {
    color: inherit;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 26px;
    background: linear-gradient(135deg, #172554, #1e3a5f);
    color: #ffffff !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
}

.hero-title,
.hero-subtitle,
.train-route {
    color: #ffffff !important;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
}

.hero-subtitle {
    margin-top: 6px;
    font-size: 18px;
    opacity: 0.9;
}

.train-route {
    margin-top: 18px;
    font-size: 16px;
}

.live-pill {
    display: inline-block;
    padding: 7px 13px;
    margin-bottom: 14px;
    border-radius: 999px;
    background: #dcfce7;
    color: #166534 !important;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.8px;
}

.section-title {
    color: #f8fafc !important;
    font-size: 21px;
    font-weight: 800;
    margin: 26px 0 14px 0;
}

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
}

.secondary-cards {
    margin-top: 14px;
}

.card {
    background: #ffffff !important;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    padding: 20px;
    min-height: 86px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.14);
}

.small-card {
    min-height: 70px;
}

.card-label {
    color: #52627a !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.8px;
}

.card-value {
    color: #172033 !important;
    margin-top: 10px;
    font-size: 20px;
    font-weight: 750;
    word-break: break-word;
}

.card-note {
    color: #64748b !important;
    margin-top: 7px;
    font-size: 12px;
    line-height: 1.35;
}

.explain-box {
    margin: 14px 0 20px 0;
    background: #eef6ff !important;
    color: #172033 !important;
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    overflow: hidden;
}

.explain-box summary {
    cursor: pointer;
    padding: 15px 18px;
    color: #1d4ed8 !important;
    font-weight: 800;
    list-style: none;
}

.explain-box summary::-webkit-details-marker {
    display: none;
}

.explain-content {
    padding: 0 18px 18px 18px;
    color: #334155 !important;
    font-size: 14px;
    line-height: 1.55;
}

.explain-content p,
.explain-content li,
.explain-content b {
    color: #334155 !important;
}

.explain-content code {
    display: inline-block;
    margin: 4px 0;
    padding: 5px 8px;
    border-radius: 7px;
    background: #dbeafe;
    color: #1e3a8a !important;
    font-family: monospace;
}

.small-explanation {
    font-size: 12px;
    color: #64748b !important;
}

.info-panel {
    margin-top: 14px;
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    padding: 17px 20px;
    display: flex;
    justify-content: space-between;
    gap: 16px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
}

.info-panel span {
    color: #172033 !important;
}

.score {
    color: #2563eb !important;
    font-size: 20px;
    font-weight: 800;
}

.prediction-card {
    background: linear-gradient(135deg, #eff6ff, #ffffff) !important;
    color: #172033 !important;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
}

.prediction-label {
    color: #52627a !important;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
}

.prediction-value {
    color: #172033 !important;
    font-size: 42px;
    font-weight: 900;
    margin-top: 8px;
}

.prediction-note {
    color: #52627a !important;
    margin-top: 10px;
    font-size: 14px;
}

.eta-panel {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid #dbe3ef;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.eta-label {
    color: #52627a !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.8px;
}

.eta-time {
    color: #172033 !important;
    margin-top: 7px;
    font-size: 20px;
    font-weight: 800;
}

.highlight {
    color: #2563eb !important;
    font-size: 24px;
}

.arrow {
    color: #64748b !important;
    font-size: 30px;
    font-weight: 900;
}

.railway-board {
    margin-top: 14px;
    padding: 0;
    border-radius: 10px;
    overflow: hidden;
    background: #171717;
    border: 2px solid #2f2f2f;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
}

.board-head,
.board-row {
    display: grid;
    grid-template-columns: 1.1fr 2.7fr 1.5fr 0.8fr 1fr;
    align-items: center;
    gap: 0;
}

.board-head {
    padding: 10px 14px;
    background: #d9d9d9;
    color: #263b8f !important;
    font-size: 11px;
    font-weight: 900;
    text-align: center;
}

.board-head span {
    color: #263b8f !important;
}

.board-row {
    padding: 16px 14px;
    background: #090909;
    color: #ff4b18 !important;
    font-family: "Courier New", monospace;
    font-size: 17px;
    font-weight: 900;
    text-align: center;
    text-shadow: 0 0 8px rgba(255, 75, 24, 0.45);
}

.board-row span {
    color: #ff4b18 !important;
}

.board-note {
    padding: 10px 14px;
    background: #202020;
    color: #bdbdbd !important;
    font-size: 11px;
    line-height: 1.4;
    text-align: center;
}

.footer {
    color: #94a3b8 !important;
    text-align: center;
    margin-top: 28px;
    padding: 18px;
    font-size: 13px;
}

@media (max-width: 700px) {
    .hero-title {
        font-size: 27px;
    }

    .eta-panel,
    .info-panel {
        flex-direction: column;
        align-items: stretch;
    }

    .arrow {
        text-align: center;
        transform: rotate(90deg);
    }
}
"""



with gr.Blocks(
    title="Smart Rail ETA",
    theme=gr.themes.Soft(),
    css=CSS,
) as app:

    gr.Markdown(
        """
# 🚆 Smart Rail ETA
### Real-time AI railway ETA prediction

**Train 12301 · Howrah → New Delhi Rajdhani Express**

Live RailRadar telemetry + OpenWeather intelligence +
trained XGBoost model
"""
    )

    refresh_button = gr.Button(
        "🔄  Refresh Live Data & Predict ETA",
        variant="primary",
        size="lg",
        elem_id="refresh-btn",
    )

    result = gr.Markdown(
        """
<div style="text-align:center; padding:50px 20px; color:#e5e7eb;">
    <h2 style="color:#f8fafc;">👋 Ready for live prediction</h2>
    <p style="color:#cbd5e1;">Click <b style="color:#ffffff;">
    Refresh Live Data & Predict ETA</b> to fetch the latest train status,
    weather and AI prediction.</p>
</div>
"""
    )

    refresh_button.click(
        fn=get_live_prediction,
        inputs=[],
        outputs=result,
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 10000)
        ),
    )
