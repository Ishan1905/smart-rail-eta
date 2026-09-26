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

artifact = joblib.load(MODEL_PATH)

xgb_model = artifact["model"]
preprocessor = artifact["preprocessor"]
feature_columns = artifact["feature_columns"]


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

    "route_historical_ontime_pct": 74.1
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
# HELPER: WEATHER → FOG RISK
# ============================================================

def calculate_fog_risk(visibility_km, humidity):

    # Dataset observation:
    # is_fog_risk = 0 -> fog_risk_score = 0
    # is_fog_risk = 1 -> approximately 0.32–0.88

    if visibility_km >= 5:

        return 0.0

    visibility_component = max(
        0,
        min(
            1,
            (5 - visibility_km) / 5
        )
    )

    humidity_component = max(
        0,
        min(
            1,
            (humidity - 70) / 30
        )
    )

    raw_score = (
        0.6 * visibility_component
        +
        0.4 * humidity_component
    )

    score = (
        0.32
        +
        raw_score * (0.88 - 0.32)
    )

    return float(
        max(
            0.32,
            min(
                0.88,
                score
            )
        )
    )


# ============================================================
# MAIN LIVE PREDICTION
# ============================================================

def get_live_prediction():

    # --------------------------------------------------------
    # Check API keys
    # --------------------------------------------------------

    if not RAILRADAR_API_KEY:

        return (
            "❌ RAILRADAR_API_KEY is missing.\n\n"
            "Add it to Render Environment Variables."
        )

    if not OPENWEATHER_API_KEY:

        return (
            "❌ OPENWEATHER_API_KEY is missing.\n\n"
            "Add it to Render Environment Variables."
        )


    # ========================================================
    # 1. RAILRADAR
    # ========================================================

    rail_url = (
        f"https://api.railradar.in/v1/trains/"
        f"{TRAIN_NUMBER}/live"
    )

    try:

        rail_response = requests.get(
            rail_url,
            headers={
                "Authorization":
                    f"Bearer {RAILRADAR_API_KEY}"
            },
            timeout=20
        )

    except Exception as e:

        return (
            "❌ RailRadar connection error:\n\n"
            f"{str(e)}"
        )


    if rail_response.status_code != 200:

        return (
            "❌ RailRadar API error\n\n"
            f"HTTP status: {rail_response.status_code}\n"
            f"{rail_response.text[:500]}"
        )


    try:

        rail_json = rail_response.json()

    except Exception:

        return "❌ RailRadar returned invalid JSON."


    if not rail_json.get("success"):

        return (
            "❌ RailRadar returned an unsuccessful response.\n\n"
            f"{rail_json}"
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


    # Find current station coordinates
    if lat is None or lon is None:

        matching = [
            station
            for station in route
            if station.get("stationCode")
            == current_code
        ]

        if matching:

            lat = matching[0].get("lat")
            lon = matching[0].get("lng")


    # Fallback to source station
    if lat is None or lon is None:

        lat = train_info["source"]["lat"]
        lon = train_info["source"]["lng"]


    current_delay = float(
        rail.get("delayMinutes", 0) or 0
    )

    speed = float(
        current.get("speedKmh", 0) or 0
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
            timeout=20
        )

    except Exception as e:

        return (
            "❌ OpenWeather connection error:\n\n"
            f"{str(e)}"
        )


    if weather_response.status_code != 200:

        return (
            "❌ OpenWeather API error\n\n"
            f"HTTP status: {weather_response.status_code}\n"
            f"{weather_response.text[:500]}"
        )


    weather = weather_response.json()


    weather_main = weather["main"]

    weather_info = weather["weather"][0]

    weather_wind = weather.get("wind", {})


    temperature = float(
        weather_main.get("temp", 0)
    )

    humidity = float(
        weather_main.get("humidity", 0)
    )

    pressure = float(
        weather_main.get("pressure", 0)
    )

    visibility_km = (
        weather.get("visibility", 10000)
        / 1000
    )

    wind_speed = float(
        weather_wind.get("speed", 0)
    )

    weather_condition = weather_info.get(
        "description",
        "unknown"
    )


    # ========================================================
    # 4. CURRENT DATE/TIME
    # ========================================================

    now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    year = now.year

    month = now.month

    day_of_week = now.weekday()

    departure_hour = now.hour


    is_weekend = int(
        day_of_week >= 5
    )

    is_night_departure = int(
        departure_hour < 6
        or departure_hour >= 22
    )

    is_peak_hour = int(
        departure_hour in [
            7, 8, 9,
            17, 18, 19, 20
        ]
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
        humidity
    )


    # ========================================================
    # 6. BUILD 36 MODEL FEATURES
    # ========================================================

    features = {

        # Categorical
        "train_number":
            12301,

        "train_type":
            "Rajdhani Express",

        "season":
            season,

        "traction_type":
            HISTORICAL_PROFILE[
                "traction_type"
            ],

        "primary_delay_cause":
            "On Time",


        # Time
        "year":
            year,

        "month":
            month,

        "day_of_week":
            day_of_week,

        "departure_hour":
            departure_hour,

        "is_weekend":
            is_weekend,

        "is_night_departure":
            is_night_departure,

        "is_peak_hour":
            is_peak_hour,

        "is_festival_season":
            is_festival_season,


        # Route
        "distance_km":
            float(
                train_info.get(
                    "distance",
                    1450.1
                )
            ),

        "num_scheduled_stops":
            int(
                train_info.get(
                    "totalHalts",
                    9
                )
            ),

        "scheduled_travel_hours":
            HISTORICAL_PROFILE[
                "scheduled_travel_hours"
            ],


        # Infrastructure
        "track_doubled":
            HISTORICAL_PROFILE[
                "track_doubled"
            ],

        "is_hdn_route":
            HISTORICAL_PROFILE[
                "is_hdn_route"
            ],

        "is_electrified":
            HISTORICAL_PROFILE[
                "is_electrified"
            ],

        "psr_count":
            HISTORICAL_PROFILE[
                "psr_count"
            ],

        "is_circular_route":
            HISTORICAL_PROFILE[
                "is_circular_route"
            ],


        # Weather / season
        "is_monsoon_season":
            is_monsoon_season,

        "fog_risk_score":
            fog_risk_score,

        "zone_fog_index":
            HISTORICAL_PROFILE[
                "zone_fog_index"
            ],

        "zone_congestion_index":
            HISTORICAL_PROFILE[
                "zone_congestion_index"
            ],

        "season_severity_score":
            HISTORICAL_PROFILE[
                "season_severity_score"
            ],


        # Train condition
        "loco_age_years":
            HISTORICAL_PROFILE[
                "loco_age_years"
            ],

        "coach_age_years":
            HISTORICAL_PROFILE[
                "coach_age_years"
            ],

        "has_lhb_coaches":
            HISTORICAL_PROFILE[
                "has_lhb_coaches"
            ],

        "is_rake_shared":
            HISTORICAL_PROFILE[
                "is_rake_shared"
            ],

        "maintenance_score":
            HISTORICAL_PROFILE[
                "maintenance_score"
            ],

        "seat_utilisation_pct":
            HISTORICAL_PROFILE[
                "seat_utilisation_pct"
            ],

        "is_overloaded":
            HISTORICAL_PROFILE[
                "is_overloaded"
            ],

        "is_special_train":
            HISTORICAL_PROFILE[
                "is_special_train"
            ],


        # Historical performance
        "route_historical_ontime_pct":
            HISTORICAL_PROFILE[
                "route_historical_ontime_pct"
            ],


        # Live delay state
        "is_delayed":
            int(current_delay > 0)
    }


    # ========================================================
    # 7. MODEL INPUT
    # ========================================================

    data = pd.DataFrame(
        [features]
    )

    data = data[
        feature_columns
    ]


    # ========================================================
    # 8. PREPROCESS + XGBOOST
    # ========================================================

    X = preprocessor.transform(
        data
    )

    predicted_delay = float(
        xgb_model.predict(X)[0]
    )

    predicted_delay = max(
        0.0,
        predicted_delay
    )


    # ========================================================
    # 9. DESTINATION + SCHEDULE
    # ========================================================

    destination = train_info[
        "destination"
    ]

    destination_code = destination[
        "code"
    ]


    destination_stop = None

    for stop in route:

        if stop.get("stationCode") == destination_code:

            destination_stop = stop
            break


    scheduled_arrival = None

    predicted_eta = None


    if destination_stop:

        scheduled_arrival = (
            destination_stop.get(
                "scheduledArrival"
            )
            or destination_stop.get(
                "arrivalTime"
            )
        )


    if scheduled_arrival:

        try:

            scheduled_arrival_dt = (
                datetime.fromisoformat(
                    scheduled_arrival.replace(
                        "Z",
                        "+00:00"
                    )
                )
            )

            scheduled_arrival_dt = (
                scheduled_arrival_dt.astimezone(
                    ZoneInfo("Asia/Kolkata")
                )
            )

            predicted_eta_dt = (
                scheduled_arrival_dt
                +
                timedelta(
                    minutes=predicted_delay
                )
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

        except Exception:

            scheduled_arrival_display = (
                str(scheduled_arrival)
            )

            predicted_eta_display = (
                "Unable to calculate"
            )

    else:

        scheduled_arrival_display = (
            "Unavailable"
        )

        predicted_eta_display = (
            "Unavailable"
        )


    # ========================================================
    # 10. USER-FACING RESULT
    # ========================================================

    status = rail.get(
        "status",
        "unknown"
    )

    current_station = current.get(
        "stationName",
        "Unknown"
    )


    result = f"""
# 🚆 Smart Rail ETA

### Train 12301
**Howrah → New Delhi Rajdhani Express**

---

### 📍 Live Train Status

| Parameter | Live Value |
|---|---|
| Current location | {current_station} |
| Train status | {status} |
| Current delay | {current_delay:.0f} min |
| Current speed | {speed:.1f} km/h |
| Coordinates | {lat:.5f}, {lon:.5f} |

---

### 🌦️ Live Weather

| Parameter | Value |
|---|---|
| Condition | {weather_condition.title()} |
| Temperature | {temperature:.1f} °C |
| Humidity | {humidity:.0f}% |
| Visibility | {visibility_km:.1f} km |
| Pressure | {pressure:.0f} hPa |
| Wind | {wind_speed:.2f} m/s |
| Fog risk score | {fog_risk_score:.2f} |

---

### 🤖 AI Prediction

**Predicted final delay: {predicted_delay:.2f} minutes**

---

### 🕐 ETA

| Parameter | Time |
|---|---|
| Scheduled arrival | {scheduled_arrival_display} |
| **Predicted ETA** | **{predicted_eta_display}** |

---

*Prediction combines the trained XGBoost model with live train status and live weather data.*
"""

    return result


# ============================================================
# WEB INTERFACE
# ============================================================

app = gr.Interface(

    fn=get_live_prediction,

    inputs=[],

    outputs=gr.Markdown(),

    title="🚆 Smart Rail ETA",

    description=(
        "Live AI-powered ETA prediction for "
        "Train 12301 using RailRadar, OpenWeather "
        "and the trained XGBoost model."
    ),
)


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )
