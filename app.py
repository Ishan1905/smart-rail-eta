import gradio as gr
import joblib
import pandas as pd

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "railway_eta_recovered_model.joblib"

artifact = joblib.load(MODEL_PATH)

xgb_model = artifact["model"]
preprocessor = artifact["preprocessor"]
feature_columns = artifact["feature_columns"]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_delay(
    train_number,
    train_type,
    year,
    month,
    day_of_week,
    departure_hour,
    is_weekend,
    is_night_departure,
    is_peak_hour,
    is_festival_season,
    season,
    distance_km,
    num_scheduled_stops,
    scheduled_travel_hours,
    track_doubled,
    is_hdn_route,
    traction_type,
    is_electrified,
    psr_count,
    is_circular_route,
    is_monsoon_season,
    fog_risk_score,
    zone_fog_index,
    zone_congestion_index,
    season_severity_score,
    loco_age_years,
    coach_age_years,
    has_lhb_coaches,
    is_rake_shared,
    maintenance_score,
    seat_utilisation_pct,
    is_overloaded,
    is_special_train,
    route_historical_ontime_pct,
    primary_delay_cause,
    is_delayed
):

    data = pd.DataFrame([{
        "train_number": train_number,
        "train_type": train_type,
        "year": year,
        "month": month,
        "day_of_week": day_of_week,
        "departure_hour": departure_hour,
        "is_weekend": is_weekend,
        "is_night_departure": is_night_departure,
        "is_peak_hour": is_peak_hour,
        "is_festival_season": is_festival_season,
        "season": season,
        "distance_km": distance_km,
        "num_scheduled_stops": num_scheduled_stops,
        "scheduled_travel_hours": scheduled_travel_hours,
        "track_doubled": track_doubled,
        "is_hdn_route": is_hdn_route,
        "traction_type": traction_type,
        "is_electrified": is_electrified,
        "psr_count": psr_count,
        "is_circular_route": is_circular_route,
        "is_monsoon_season": is_monsoon_season,
        "fog_risk_score": fog_risk_score,
        "zone_fog_index": zone_fog_index,
        "zone_congestion_index": zone_congestion_index,
        "season_severity_score": season_severity_score,
        "loco_age_years": loco_age_years,
        "coach_age_years": coach_age_years,
        "has_lhb_coaches": has_lhb_coaches,
        "is_rake_shared": is_rake_shared,
        "maintenance_score": maintenance_score,
        "seat_utilisation_pct": seat_utilisation_pct,
        "is_overloaded": is_overloaded,
        "is_special_train": is_special_train,
        "route_historical_ontime_pct": route_historical_ontime_pct,
        "primary_delay_cause": primary_delay_cause,
        "is_delayed": is_delayed
    }])

    # Ensure exact feature order
    data = data[feature_columns]

    # Apply saved preprocessing
    X = preprocessor.transform(data)

    # Predict delay
    prediction = xgb_model.predict(X)[0]

    return f"{float(prediction):.2f} minutes"


# ============================================================
# WEB INTERFACE
# ============================================================

app = gr.Interface(
    fn=predict_delay,

    inputs=[
        gr.Number(value=12301, label="Train Number"),
        gr.Textbox(value="Superfast", label="Train Type"),
        gr.Number(value=2024, label="Year"),
        gr.Number(value=12, label="Month"),
        gr.Number(value=1, label="Day of Week"),
        gr.Number(value=10, label="Departure Hour"),
        gr.Number(value=0, label="Weekend (0/1)"),
        gr.Number(value=0, label="Night Departure (0/1)"),
        gr.Number(value=1, label="Peak Hour (0/1)"),
        gr.Number(value=0, label="Festival Season (0/1)"),
        gr.Textbox(value="Winter", label="Season"),
        gr.Number(value=1000, label="Distance (km)"),
        gr.Number(value=20, label="Scheduled Stops"),
        gr.Number(value=14, label="Scheduled Travel Hours"),
        gr.Number(value=1, label="Track Doubled (0/1)"),
        gr.Number(value=1, label="HDN Route (0/1)"),
        gr.Textbox(value="Electric", label="Traction Type"),
        gr.Number(value=1, label="Electrified (0/1)"),
        gr.Number(value=2, label="PSR Count"),
        gr.Number(value=0, label="Circular Route (0/1)"),
        gr.Number(value=0, label="Monsoon (0/1)"),
        gr.Number(value=0.2, label="Fog Risk Score"),
        gr.Number(value=0.2, label="Zone Fog Index"),
        gr.Number(value=0.5, label="Zone Congestion Index"),
        gr.Number(value=0.2, label="Season Severity Score"),
        gr.Number(value=5, label="Loco Age (years)"),
        gr.Number(value=5, label="Coach Age (years)"),
        gr.Number(value=1, label="LHB Coaches (0/1)"),
        gr.Number(value=0, label="Rake Shared (0/1)"),
        gr.Number(value=0.9, label="Maintenance Score"),
        gr.Number(value=75, label="Seat Utilisation (%)"),
        gr.Number(value=0, label="Overloaded (0/1)"),
        gr.Number(value=0, label="Special Train (0/1)"),
        gr.Number(value=85, label="Historical On-Time (%)"),
        gr.Textbox(value="Operational", label="Primary Delay Cause"),
        gr.Number(value=0, label="Is Delayed (0/1)")
    ],

    outputs=gr.Textbox(label="Predicted Delay"),

    title="🚆 Smart Rail ETA",
    description="AI-powered railway delay prediction using the trained XGBoost model."
)


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    app.launch()
