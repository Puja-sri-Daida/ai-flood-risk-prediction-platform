import os
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import streamlit as st

from shapely.geometry import Point


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Disaster Intelligence Platform",
    page_icon="🌊",
    layout="wide"
)
# ============================================================
# LOGIN / USER ACCESS
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


if not st.session_state.authenticated:

    st.title("🌊 AI Disaster Intelligence & Early Warning Platform")

    st.subheader("🔐 User Login")

    st.write(
        "Login to access the AI-powered flood risk assessment dashboard."
    )

    username = st.text_input(
        "Username",
        placeholder="Enter username"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password"
    )

    login_button = st.button(
        "🔐 Login",
        width="stretch"
    )

    if login_button:

        correct_username = st.secrets["auth"]["username"]
        correct_password = st.secrets["auth"]["password"]

        if (
            username == correct_username
            and password == correct_password
        ):

            st.session_state.authenticated = True

            st.success(
                "✅ Login successful. Welcome!"
            )

            st.rerun()

        else:

            st.error(
                "❌ Invalid username or password."
            )

    st.stop()


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "flood_model.joblib"
)

RIVER_FILE = os.path.join(BASE_DIR, "data", "india_rivers.fgb")

FLOOD_FILE = os.path.join(
    BASE_DIR,
    "global_flood_records.gpkg"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "rainfall_24h_mm",
    "rainfall_3day_mm",
    "humidity_pct",
    "temperature_c",
    "elevation_m",
    "slope_deg",
    "river_distance_km",
    "soil_moisture_pct",
    "historical_flood_index"
]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model_data = joblib.load(MODEL_PATH)

    if not isinstance(model_data, dict):
        raise ValueError(
            "The model file does not contain the expected dictionary structure."
        )

    if "model" not in model_data:
        raise ValueError(
            "The model dictionary does not contain 'model'."
        )

    model = model_data["model"]

    saved_features = model_data.get(
        "features",
        FEATURES
    )

    return model, saved_features


# ============================================================
# LOCATION SEARCH
# ============================================================

def get_location(place):
    aliases = {
        "vizag": "Visakhapatnam",
        "bombay": "Mumbai",
        "calcutta": "Kolkata",
        "madras": "Chennai",
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "hyderabad": "Hyderabad",
        "secunderabad": "Secunderabad",
        "delhi": "Delhi",
        "new delhi": "New Delhi",
        "trivandrum": "Thiruvananthapuram",
        "cochin": "Kochi",
        "pondicherry": "Puducherry",
        "gurgaon": "Gurugram",
        "baroda": "Vadodara",
        "calicut": "Kozhikode",
    }

    search_name = place.strip()

    if not search_name:
        raise ValueError(
            "Please enter a valid area or city name."
        )

    alias_key = search_name.lower()

    if alias_key in aliases:
        search_name = aliases[alias_key]

    # --------------------------------------------------------
    # PRIMARY GEOCODER: OPEN-METEO
    # --------------------------------------------------------

    open_meteo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    open_meteo_params = {
        "name": search_name,
        "count": 10,
        "language": "en",
        "format": "json",
        "countryCode": "IN",
    }

    response = requests.get(
        open_meteo_url,
        params=open_meteo_params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if results:
        india_results = [
            result
            for result in results
            if result.get("country_code", "").upper() == "IN"
        ]

        if india_results:
            result = india_results[0]
        else:
            result = results[0]

        return {
            "name": result["name"],
            "country": result.get("country", "India"),
            "latitude": float(result["latitude"]),
            "longitude": float(result["longitude"]),
            "elevation": result.get("elevation"),
        }

    # --------------------------------------------------------
    # FALLBACK GEOCODER: OPENSTREETMAP NOMINATIM
    # --------------------------------------------------------

    nominatim_url = (
        "https://nominatim.openstreetmap.org/search"
    )

    nominatim_params = {
        "q": f"{search_name}, India",
        "format": "json",
        "limit": 10,
        "addressdetails": 1,
    }

    headers = {
        "User-Agent": (
            "AI-Disaster-Intelligence-App/1.0 "
            "(flood-risk-analysis)"
        )
    }

    fallback_response = requests.get(
        nominatim_url,
        params=nominatim_params,
        headers=headers,
        timeout=20
    )

    fallback_response.raise_for_status()

    fallback_results = fallback_response.json()

    if not fallback_results:
        raise ValueError(
            f"Location '{place}' was not found. "
            "Please enter a valid city, town, village, "
            "or area name."
        )

    # Prefer actual populated places over administrative boundaries.
    place_types = {
        "city",
        "town",
        "village",
        "hamlet",
        "suburb",
        "neighbourhood",
        "quarter",
        "isolated_dwelling",
    }

    valid_places = [
        result
        for result in fallback_results
        if (
            result.get("type", "").lower()
            in place_types
        )
    ]

    if not valid_places:
        valid_places = fallback_results

    # Prefer exact name matches.
    exact_matches = [
        result
        for result in valid_places
        if result.get("name", "").strip().lower()
        == search_name.strip().lower()
    ]

    if exact_matches:
        valid_places = exact_matches

    result = valid_places[0]

    address = result.get("address", {})

    return {
        "name": result.get(
            "name",
            search_name
        ),
        "country": address.get(
            "country",
            "India"
        ),
        "latitude": float(
            result["lat"]
        ),
        "longitude": float(
            result["lon"]
        ),
        "elevation": None,
    }
# ============================================================
# WEATHER + ENVIRONMENT DATA
# ============================================================

def get_environmental_data(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "soil_moisture_0_to_1cm"
        ]),

        "hourly": "precipitation",

        "past_days": 3,
        "forecast_days": 1,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    current = data["current"]

    temperature = current.get(
        "temperature_2m"
    )

    humidity = current.get(
        "relative_humidity_2m"
    )

    soil_fraction = current.get(
        "soil_moisture_0_to_1cm"
    )

    if soil_fraction is None:
        soil_moisture_pct = 0.0
    else:
        soil_moisture_pct = float(
            soil_fraction
        ) * 100

    hourly_times = data["hourly"]["time"]
    rainfall_values = data["hourly"]["precipitation"]

    current_time = current["time"]

    completed_rainfall = []

    for time_value, rainfall_value in zip(
        hourly_times,
        rainfall_values
    ):

        if time_value <= current_time:

            if rainfall_value is not None:
                completed_rainfall.append(
                    float(rainfall_value)
                )

    rainfall_24h = sum(
        completed_rainfall[-24:]
    )

    rainfall_3day = sum(
        completed_rainfall[-72:]
    )

    elevation = data.get("elevation")

    if elevation is None:
        elevation = 0.0

    return {
        "rainfall_24h_mm": rainfall_24h,
        "rainfall_3day_mm": rainfall_3day,
        "humidity_pct": float(humidity),
        "temperature_c": float(temperature),
        "elevation_m": float(elevation),
        "soil_moisture_pct": soil_moisture_pct
    }


# ============================================================
# UTM ZONE
# ============================================================

def get_utm_epsg(latitude, longitude):

    utm_zone = int(
        (longitude + 180) / 6
    ) + 1

    if latitude >= 0:
        return 32600 + utm_zone

    return 32700 + utm_zone


# ============================================================
# AUTOMATIC SLOPE
# ============================================================

def get_slope(latitude, longitude):

    url = "https://api.open-meteo.com/v1/elevation"

    offset = 0.001

    latitudes = [
        latitude,
        latitude + offset,
        latitude - offset
    ]

    longitudes = [
        longitude,
        longitude,
        longitude
    ]

    params = {
        "latitude": ",".join(
            map(str, latitudes)
        ),
        "longitude": ",".join(
            map(str, longitudes)
        )
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    elevations = data["elevation"]

    north = float(elevations[1])
    south = float(elevations[2])

    horizontal_distance = 222.0

    elevation_difference = (
        north - south
    )

    slope_deg = math.degrees(
        math.atan(
            abs(elevation_difference)
            / horizontal_distance
        )
    )

    return slope_deg


# ============================================================
# AUTOMATIC RIVER DISTANCE
# ============================================================

def get_river_distance(latitude, longitude):

    if not os.path.exists(RIVER_FILE):
        return None

    buffer = 0.5

    bbox = (
        longitude - buffer,
        latitude - buffer,
        longitude + buffer,
        latitude + buffer
    )

    rivers = gpd.read_file(
        RIVER_FILE,
        bbox=bbox
    )

    if rivers.empty:
        return None

    rivers = rivers[
        rivers.geometry.notna()
    ]

    rivers = rivers[
        ~rivers.geometry.is_empty
    ]

    if rivers.empty:
        return None

    location = gpd.GeoDataFrame(
        {"name": ["User Location"]},
        geometry=[
            Point(
                longitude,
                latitude
            )
        ],
        crs="EPSG:4326"
    )

    epsg = get_utm_epsg(
        latitude,
        longitude
    )

    rivers_projected = rivers.to_crs(
        epsg=epsg
    )

    location_projected = location.to_crs(
        epsg=epsg
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            RuntimeWarning
        )

        distances = (
            rivers_projected.geometry.distance(
                location_projected.geometry.iloc[0]
            )
        )

    distances = distances[
        np.isfinite(distances)
    ]

    if distances.empty:
        return None

    nearest_distance_m = distances.min()

    return float(
        nearest_distance_m / 1000
    )


# ============================================================
# HISTORICAL FLOOD INDEX
# ============================================================

def get_historical_flood_index(
    latitude,
    longitude
):

    if not os.path.exists(FLOOD_FILE):
        return 0.0

    floods = gpd.read_file(
        FLOOD_FILE
    )

    if "FloodImpactIndex" not in floods.columns:
        return 0.0

    floods["FloodImpactIndex"] = pd.to_numeric(
        floods["FloodImpactIndex"],
        errors="coerce"
    )

    floods = floods[
        floods["FloodImpactIndex"].notna()
    ]

    floods = floods[
        floods.geometry.notna()
    ]

    floods = floods[
        ~floods.geometry.is_empty
    ]

    floods = floods[
        floods.geometry.is_valid
    ]

    if floods.empty:
        return 0.0

    location = gpd.GeoSeries(
        [
            Point(
                longitude,
                latitude
            )
        ],
        crs="EPSG:4326"
    )

    epsg = get_utm_epsg(
        latitude,
        longitude
    )

    floods_projected = floods.to_crs(
        epsg=epsg
    )

    location_projected = location.to_crs(
        epsg=epsg
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            RuntimeWarning
        )

        distances = (
            floods_projected.geometry.distance(
                location_projected.iloc[0]
            )
        )

    distances = np.asarray(
        distances,
        dtype=float
    )

    valid_distance = np.isfinite(
        distances
    )

    floods = floods.loc[
        valid_distance
    ].copy()

    distances = distances[
        valid_distance
    ]

    if len(floods) == 0:
        return 0.0

    floods["distance_km"] = (
        distances / 1000
    )

    nearby = floods[
        floods["distance_km"] <= 50
    ]

    if nearby.empty:
        return 0.0

    historical_index = nearby[
        "FloodImpactIndex"
    ].mean()

    # Convert DFO index (approximately 0–8.49)
    # into the 0–1 scale used by the trained model.

    historical_index = (
        historical_index / 8.49
    )

    historical_index = min(
        max(
            float(historical_index),
            0.0
        ),
        1.0
    )

    return historical_index


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(
    probability,
    prediction
):

    if prediction == 1:

        if probability >= 0.75:
            return "CRITICAL"

        if probability >= 0.50:
            return "HIGH"

        return "MODERATE"

    if probability >= 0.50:
        return "HIGH"

    if probability >= 0.25:
        return "MODERATE"

    return "LOW"


# ============================================================
# RISK COLOR / MESSAGE
# ============================================================

def get_risk_message(risk_level):

    messages = {

        "CRITICAL":
            "Very high flood-risk conditions detected. Take immediate precautions.",

        "HIGH":
            "High flood-risk conditions detected. Stay alert and follow official alerts.",

        "MODERATE":
            "Moderate flood-risk conditions detected. Continue monitoring weather conditions.",

        "LOW":
            "Current model conditions indicate relatively low flood risk."
    }

    return messages.get(
        risk_level,
        "Monitor local conditions."
    )


# ============================================================
# SAFETY RECOMMENDATIONS
# ============================================================

def get_recommendations(
    risk_level
):

    recommendations = {

        "CRITICAL": [
            "Move to a safer or elevated location if local authorities advise evacuation.",
            "Follow official government and disaster-management alerts.",
            "Avoid walking or driving through flood water.",
            "Keep medicines, documents, water and emergency supplies ready.",
            "Avoid flooded electrical areas and exposed electrical equipment."
        ],

        "HIGH": [
            "Monitor official weather and flood alerts.",
            "Prepare an evacuation plan.",
            "Keep emergency supplies ready.",
            "Avoid unnecessary travel near rivers and low-lying areas.",
            "Keep your phone charged."
        ],

        "MODERATE": [
            "Continue monitoring rainfall and weather conditions.",
            "Keep emergency contacts available.",
            "Be careful around low-lying and poorly drained areas.",
            "Keep basic emergency supplies ready."
        ],

        "LOW": [
            "Continue monitoring local weather conditions.",
            "Keep emergency alerts enabled.",
            "Stay aware of nearby flood-prone areas."
        ]
    }

    return recommendations.get(
        risk_level,
        []
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🌊 AI Disaster Intelligence & Early Warning Platform"
)

st.markdown(
    """
    ### 🌍 Automated Flood Risk Assessment

    Enter an **area or city** to generate a location-based flood risk
    assessment using automatically collected weather, terrain,
    river-network, and historical flood information.
    """
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📍 Flood Risk Analysis"
)
place = st.sidebar.text_input(
    "Enter area / city",
    placeholder="Example: Hyderabad",
    help="Enter an Indian city, town, or area to analyze flood risk."
)
st.sidebar.caption("🇮🇳 Supports Indian cities, towns, and areas")


analyze = st.sidebar.button(
    "🔍 Analyze Location",
    width="stretch"
)


st.sidebar.markdown("---")

st.sidebar.info(
    "The system automatically collects rainfall, "
    "humidity, temperature, elevation, soil moisture, "
    "slope, river distance and historical flood information."
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze:

    if not place.strip():

        st.warning(
            "Please enter an area or city."
        )

        st.stop()

    try:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        with st.spinner(
            "Loading trained flood prediction model..."
        ):

            model, model_features = load_model()


        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        with st.spinner(
            "Finding location..."
        ):

            location = get_location(
                place.strip()
            )

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # AUTOMATIC ENVIRONMENT DATA
        # ----------------------------------------------------

        with st.spinner(
            "🌦 Collecting environmental data..."
        ):

            environment = (
                get_environmental_data(
                    latitude,
                    longitude
                )
            )


        # ----------------------------------------------------
        # LOCATION DISPLAY
        # ----------------------------------------------------

        st.subheader(
            "📍 Location Found"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Area",
                location["name"]
            )

        with col2:

            st.metric(
                "Country",
                location["country"]
            )

        with col3:

            st.metric(
                "Elevation",
                f"{environment['elevation_m']:.1f} m"
            )


        st.caption(
            f"Coordinates: {latitude:.5f}, {longitude:.5f}"
        )


        # ----------------------------------------------------
        # AUTOMATIC SLOPE
        # ----------------------------------------------------

        with st.spinner(
            "⛰ Calculating terrain slope..."
        ):

            slope = get_slope(
                latitude,
                longitude
            )


        # ----------------------------------------------------
        # AUTOMATIC RIVER DISTANCE
        # ----------------------------------------------------

        with st.spinner(
            "🌊 Calculating nearest river distance..."
        ):

            river_distance = (
                get_river_distance(
                    latitude,
                    longitude
                )
            )

        if river_distance is None:

            river_distance = 999.0


        # ----------------------------------------------------
        # HISTORICAL FLOOD DATA
        # ----------------------------------------------------

        with st.spinner(
            "📚 Checking historical flood records..."
        ):

            historical_index = (
                get_historical_flood_index(
                    latitude,
                    longitude
                )
            )


        # ----------------------------------------------------
        # ALL 9 FEATURES
        # ----------------------------------------------------

        features = {

            "rainfall_24h_mm":
                environment[
                    "rainfall_24h_mm"
                ],

            "rainfall_3day_mm":
                environment[
                    "rainfall_3day_mm"
                ],

            "humidity_pct":
                environment[
                    "humidity_pct"
                ],

            "temperature_c":
                environment[
                    "temperature_c"
                ],

            "elevation_m":
                environment[
                    "elevation_m"
                ],

            "slope_deg":
                slope,

            "river_distance_km":
                river_distance,

            "soil_moisture_pct":
                environment[
                    "soil_moisture_pct"
                ],

            "historical_flood_index":
                historical_index
        }


        # ----------------------------------------------------
        # MODEL INPUT DATAFRAME
        # ----------------------------------------------------

        input_values = []

        for feature_name in model_features:

            if feature_name not in features:

                raise ValueError(
                    f"Model feature '{feature_name}' "
                    "is missing from collected data."
                )

            input_values.append(
                features[feature_name]
            )

        input_df = pd.DataFrame(
            [input_values],
            columns=model_features
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Running AI flood-risk prediction..."
        ):

            prediction = int(
                np.asarray(
                    model.predict(input_df)
                ).ravel()[0]
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    input_df
                )[0]
            )

            class_values = list(
                model.classes_
            )

            if 1 in class_values:

                class_1_index = (
                    class_values.index(1)
                )

                probability = float(
                    probabilities[
                        class_1_index
                    ]
                )

            else:

                probability = 1.0 if prediction == 1 else 0.0

        else:

            probability = (
                1.0
                if prediction == 1
                else 0.0
            )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk_level = calculate_risk_level(
            probability,
            prediction
        )

        risk_message = get_risk_message(
            risk_level
        )# ----------------------------------------------------
        # FLOOD RISK RESULT
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader("🚨 Flood Risk Prediction")

        # ----------------------------------------------------
        # MAIN RISK STATUS
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ HIGH FLOOD RISK DETECTED"
            )

        else:

            st.success(
                "✅ LOW FLOOD RISK DETECTED"
            )

        # ----------------------------------------------------
        # RISK METRICS
        # ----------------------------------------------------

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:

            st.metric(
                "🌊 Model Prediction",
                "HIGH" if prediction == 1 else "LOW"
            )

        with metric_col2:

            st.metric(
                "📊 Risk Probability",
                f"{probability * 100:.1f}%"
            )

        with metric_col3:

            st.metric(
                "⚠️ Risk Assessment",
                risk_level
            )

        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        st.markdown("### 📈 Flood Risk Score")

        risk_score = round(
            probability * 100
        )

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )

        st.caption(
            f"Current model risk score: "
            f"**{risk_score} / 100**"
        )

        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk_level == "CRITICAL":

            st.error(
                f"🚨 {risk_message}"
            )

        elif risk_level == "HIGH":

            st.warning(
                f"⚠️ {risk_message}"
            )

        elif risk_level == "MODERATE":

            st.warning(
                f"🟡 {risk_message}"
            )

        else:

            st.success(
                f"🟢 {risk_message}"
            )



        # ----------------------------------------------------
        # MODEL PERFORMANCE
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("📊 Model Performance")

        st.caption(
            "Evaluation of the recreated XGBoost model on a stratified "
            "20% hold-out split of the project dataset."
        )

        # Evaluation metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric(
                "Accuracy",
                "82.58%"
            )

        with metric_col2:
            st.metric(
                "Precision",
                "85.51%"
            )

        with metric_col3:
            st.metric(
                "Recall",
                "86.99%"
            )

        with metric_col4:
            st.metric(
                "F1-Score",
                "86.24%"
            )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.markdown("### 🔢 Confusion Matrix")

        confusion_matrix_data = pd.DataFrame(
            [
                [336, 111],
                [98, 655],
            ],
            index=["Actual No Flood", "Actual Flood"],
            columns=["Predicted No Flood", "Predicted Flood"],
        )

        st.dataframe(
            confusion_matrix_data,
            width="stretch"
        )

        st.caption(
            "Evaluation set: 1,200 samples from a stratified 20% hold-out split."
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.markdown("### 🌳 Feature Importance")

        if hasattr(model, "feature_importances_"):

            importance_data = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Importance": model.feature_importances_,
                }
            )

            importance_data = importance_data.sort_values(
                by="Importance",
                ascending=False
            )

            importance_data = importance_data.set_index(
                "Feature"
            )

            st.bar_chart(
                importance_data,
                width="stretch"
            )

        else:

            st.info(
                "Feature importance is not available for this model."
            )

        st.markdown("### 📋 Evaluation Summary")

        st.write(
            """
            The recreated XGBoost model was evaluated using a stratified
            20% hold-out split of the project dataset. The evaluation
            measured accuracy, precision, recall, and F1-score for flood
            prediction.
            """
        )

        st.info(
            "⚠️ These metrics represent performance on the project "
            "dataset and should not be interpreted as guaranteed "
            "real-world prediction accuracy."
        )
        # ----------------------------------------------------
        # HISTORICAL FLOOD DATA METHODOLOGY
        # ----------------------------------------------------

        with st.expander("🌊 Historical Flood Data & Methodology"):

            st.markdown(
                """
                **How the historical flood index is calculated**

                1. Historical flood records are loaded from the project's
                   global flood dataset.

                2. Flood records within **50 km** of the selected location
                   are identified using geographic distance calculations.

                3. The `FloodImpactIndex` values of the nearby historical
                   events are averaged.

                4. The resulting value is normalized to the **0–1 scale**
                   required by the machine learning model.

                5. If no historical flood records are found within 50 km,
                   the historical flood index is set to `0.0`.
                """
            )

            st.info(
                "ℹ️ The 0–1 normalization is a project-level engineering "
                "choice based on the observed maximum FloodImpactIndex "
                "of approximately 8.49 in the historical dataset. "
                "It is not an official DFO probability or threshold."
            )

            st.caption(
                "Historical flood information is used as one input feature "
                "along with weather, terrain, soil, and river-network data."
            )

# ================= INTERACTIVE MAP =================

        st.subheader("📍 Location Map")

        map_data = pd.DataFrame(
            {
                "latitude": [location["latitude"]],
                "longitude": [location["longitude"]],
            }
        )

        st.map(
            map_data,
            latitude="latitude",
            longitude="longitude",
            zoom=10,
            size=200,
        )

        st.caption(
            f"📍 {location['name']}, {location['country']} — "
            f"{location['latitude']:.4f}, {location['longitude']:.4f}"
        )

        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        if risk_level == "CRITICAL":


            st.error(
                risk_message
            )

        elif risk_level == "HIGH":

            st.warning(
                risk_message
            )

        elif risk_level == "MODERATE":

            st.warning(
                risk_message
            )

        else:

            st.success(
                risk_message
            )


        # ----------------------------------------------------
        # AUTOMATIC INPUTS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🌦 Automatically Collected Environmental Data"
        )

        display_data = pd.DataFrame(
            {
                "Parameter": [
                    "Rainfall - 24 hours",
                    "Rainfall - 3 days",
                    "Humidity",
                    "Temperature",
                    "Elevation",
                    "Terrain slope",
                    "Nearest river",
                    "Soil moisture",
                    "Historical flood index"
                ],

                "Value": [
                    f"{features['rainfall_24h_mm']:.2f} mm",
                    f"{features['rainfall_3day_mm']:.2f} mm",
                    f"{features['humidity_pct']:.1f} %",
                    f"{features['temperature_c']:.1f} °C",
                    f"{features['elevation_m']:.1f} m",
                    f"{features['slope_deg']:.2f}°",
                    f"{features['river_distance_km']:.2f} km",
                    f"{features['soil_moisture_pct']:.1f} %",
                    f"{features['historical_flood_index']:.2f}"
                ]
            }
        )

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )


        # ----------------------------------------------------
        # SAFETY RECOMMENDATIONS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🛡️ Safety Recommendations"
        )

        recommendations = (
            get_recommendations(
                risk_level
            )
        )

        for recommendation in recommendations:

            st.markdown(
                f"• {recommendation}"
            )


        # ----------------------------------------------------
        # MODEL INPUT DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Model Inputs"
        ):

            model_input_table = pd.DataFrame(
                {
                    "Feature": model_features,
                    "Value": [
                        features[name]
                        for name in model_features
                    ]
                }
            )

            st.dataframe(
                model_input_table,
                width="stretch",
                hide_index=True
            )


        # ----------------------------------------------------
        # DATA NOTES
        # ----------------------------------------------------

        with st.expander(
            "ℹ️ How This Prediction Is Generated"
        ):

            st.markdown(
                """
                **Weather and environmental data**
                are collected through Open-Meteo APIs.

                **River distance**
                is calculated using the HydroRIVERS geographic dataset.

                **Historical flood information**
                is obtained from the project's historical flood records.

                **Terrain slope**
                is estimated from elevation points around the selected location.

                The final nine environmental/geographical features
                are passed to the trained Machine Learning model.
                """
            )


    except requests.RequestException as error:

        st.error(
            f"Network/API error: {error}"
        )

    except FileNotFoundError as error:

        st.error(
            str(error)
        )

    except Exception as error:

        st.error(
            f"Something went wrong: {error}"
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👈 Enter your area/city in the sidebar "
        "and click **Analyze Flood Risk**."
    )

    st.markdown(
        """
        ### 🔄 Automatic Workflow

        **1️⃣ Enter Location**  
        ↓  
        **2️⃣ Geocode Location**  
        ↓  
        **3️⃣ Collect Weather Data**  
        ↓  
        **4️⃣ Calculate Terrain & River Information**  
        ↓  
        **5️⃣ Check Historical Flood Records**  
        ↓  
        **6️⃣ Collect All 9 ML Features**  
        ↓  
        **7️⃣ Run Machine Learning Model**  
        ↓  
        **8️⃣ Display Flood Risk & Safety Recommendations**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌊 AI Disaster Intelligence & Early Warning Platform"
)

st.caption(
    "⚠️ This platform provides AI-based risk assessment for "
    "decision support and does not replace official disaster alerts."
)
