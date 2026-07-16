import csv
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression



# ================== Page setup ==================
st.set_page_config(
    page_title="Weather Forecasting System",
    page_icon="🌦️",
    layout="wide",
)


# ================== Load city data ==================
@st.cache_data
def load_city_coordinates(file_path: str) -> dict:
    """Load country/city coordinates from worldcities.csv."""
    city_coordinates = {}

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        required_columns = {"country", "city", "lat", "lng"}
        if not required_columns.issubset(reader.fieldnames or []):
            missing = required_columns.difference(reader.fieldnames or [])
            raise ValueError(
                "worldcities.csv is missing these columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            try:
                country_name = row["country"].strip()
                city_name = row["city"].strip()
                latitude = float(row["lat"])
                longitude = float(row["lng"])
            except (TypeError, ValueError, KeyError):
                continue

            if not country_name or not city_name:
                continue

            city_coordinates.setdefault(country_name, {})
            city_coordinates[country_name][city_name] = (
                latitude,
                longitude,
            )

    return city_coordinates


# ================== Fetch weather data ==================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch hourly historical weather data from Open-Meteo."""
    api_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),
        "timezone": "auto",
    }

    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo returned an error."))

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("The weather API response does not contain hourly data.")

    required_fields = {
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
    }
    missing_fields = required_fields.difference(hourly)

    if missing_fields:
        raise ValueError(
            "The weather API response is missing: "
            + ", ".join(sorted(missing_fields))
        )

    weather_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"], errors="coerce"),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    )

    return weather_df


# ================== User interface ==================
lang = st.sidebar.selectbox(
    "اللغة / Language",
    ["English", "العربية"],
)
is_ar = lang == "العربية"

st.title(
    "نظام تنبؤ بالطقس باستخدام التعلم الآلي"
    if is_ar
    else "Weather forecasting system using machine learning"
)

try:
    city_coords = load_city_coordinates("worldcities.csv")
except FileNotFoundError:
    st.error(
        "تعذر العثور على ملف worldcities.csv."
        if is_ar
        else "worldcities.csv was not found. Put it in the same folder as app.py."
    )
    st.stop()
except (OSError, ValueError) as error:
    st.error(
        f"حدث خطأ أثناء قراءة ملف المدن: {error}"
        if is_ar
        else f"Could not read worldcities.csv: {error}"
    )
    st.stop()

if not city_coords:
    st.error(
        "لا يحتوي ملف المدن على بيانات صالحة."
        if is_ar
        else "worldcities.csv does not contain valid city data."
    )
    st.stop()

st.sidebar.markdown(
    "### اختر الدولة والمدينة"
    if is_ar
    else "### Select Country and City"
)

countries = sorted(city_coords.keys())
country = st.sidebar.selectbox(
    "الدولة" if is_ar else "Country",
    countries,
)

cities = sorted(city_coords[country].keys())
city = st.sidebar.selectbox(
    "المدينة" if is_ar else "City",
    cities,
)

lat, lon = city_coords[country][city]

st.sidebar.markdown(
    "### ماذا تريد أن يتم التنبؤ به؟"
    if is_ar
    else "### Select what to predict"
)

display_to_variable = {
    ("درجة الحرارة" if is_ar else "Temperature"): "temperature",
    ("الرطوبة" if is_ar else "Humidity"): "humidity",
    ("سرعة الرياح" if is_ar else "Wind Speed"): "wind_speed",
}

selected_display = st.sidebar.multiselect(
    "المتغيرات" if is_ar else "Variables",
    list(display_to_variable.keys()),
    default=list(display_to_variable.keys()),
)

selected_vars = [
    display_to_variable[item]
    for item in selected_display
]

st.sidebar.markdown(
    "### اختر وحدات القياس"
    if is_ar
    else "### Select units"
)

# Internal values stay language-independent.
temperature_unit = st.sidebar.radio(
    "درجة الحرارة" if is_ar else "Temperature",
    ["C", "F"],
    format_func=lambda value: (
        "°م" if value == "C" else "°ف"
    ) if is_ar else (
        "°C" if value == "C" else "°F"
    ),
    index=0,
)

wind_unit = st.sidebar.radio(
    "سرعة الرياح" if is_ar else "Wind Speed",
    ["kmh", "ms"],
    format_func=lambda value: (
        "كم/سا" if value == "kmh" else "م/ث"
    ) if is_ar else (
        "km/h" if value == "kmh" else "m/s"
    ),
    index=0,
)

start_prediction = st.sidebar.button(
    "ابدأ التنبؤ" if is_ar else "Start Prediction",
    type="primary",
    use_container_width=True,
)


# ================== Prediction pipeline ==================
if start_prediction:
    if not selected_vars:
        st.warning(
            "اختر متغيراً واحداً على الأقل."
            if is_ar
            else "Select at least one variable to predict."
        )
        st.stop()

    today = date.today()
    start_date = (today - timedelta(days=730)).isoformat()
    end_date = today.isoformat()

    loading_text = (
        "جارٍ تحميل بيانات الطقس وتدريب النماذج..."
        if is_ar
        else "Loading weather data and training models..."
    )

    try:
        with st.spinner(loading_text):
            df = fetch_historical_weather(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
            )

    except requests.Timeout:
        st.error(
            "انتهت مهلة الاتصال بواجهة الطقس. حاول مرة أخرى."
            if is_ar
            else "The weather API request timed out. Please try again."
        )
        st.stop()

    except requests.RequestException as error:
        st.error(
            f"فشل الاتصال بواجهة الطقس: {error}"
            if is_ar
            else f"Failed to connect to the weather API: {error}"
        )
        st.stop()

    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        st.error(
            f"بيانات الطقس المستلمة غير صالحة: {error}"
            if is_ar
            else f"The weather API returned invalid data: {error}"
        )
        st.stop()

    # ---------- Clean and fill missing values ----------
    weather_columns = ["temperature", "humidity", "wind_speed"]

    df = (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    for column in weather_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        # Linear interpolation fills gaps between valid observations.
        df[column] = df[column].interpolate(
            method="linear",
            limit_direction="both",
        )

        # Pandas 3-compatible replacement for:
        # fillna(method="ffill").fillna(method="bfill")
        df[column] = df[column].ffill().bfill()

    if df.empty:
        st.error(
            "لم يتم العثور على بيانات طقس قابلة للاستخدام."
            if is_ar
            else "No usable weather data was found."
        )
        st.stop()

    if df[selected_vars].isna().any().any():
        st.error(
            "توجد أعمدة لا تحتوي على بيانات كافية للتنبؤ."
            if is_ar
            else "Some selected columns do not contain enough usable data."
        )
        st.stop()

    # ---------- Train models and predict ----------
    look_back = 72
    hours_ahead = 24
    forecast_results = {}

    if len(df) <= look_back:
        st.error(
            "لا توجد بيانات كافية لتدريب النموذج."
            if is_ar
            else "There is not enough data to train the model."
        )
        st.stop()

    for variable in selected_vars:
        values = df[variable].to_numpy(dtype=float)

        X = []
        y = []

        for index in range(len(values) - look_back):
            X.append(values[index:index + look_back])
            y.append(values[index + look_back])

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.size == 0 or y.size == 0:
            st.error(
                f"لا توجد بيانات كافية لتدريب نموذج {variable}."
                if is_ar
                else f"Not enough data to train the {variable} model."
            )
            st.stop()

        model = LinearRegression()
        model.fit(X, y)

        current_sequence = values[-look_back:].reshape(1, -1)
        hourly_predictions = []

        for _ in range(hours_ahead):
            prediction = float(model.predict(current_sequence)[0])

            # Keep physically constrained variables in valid ranges.
            if variable == "humidity":
                prediction = float(np.clip(prediction, 0, 100))
            elif variable == "wind_speed":
                prediction = max(0.0, prediction)

            hourly_predictions.append(prediction)

            current_sequence = np.concatenate(
                [
                    current_sequence[:, 1:],
                    np.array([[prediction]], dtype=float),
                ],
                axis=1,
            )

        forecast_results[variable] = hourly_predictions

    # ---------- Prepare tomorrow's forecast table ----------
    forecast_day = today + timedelta(days=1)
    forecast_start = datetime.combine(
        forecast_day,
        datetime.min.time(),
    )

    hourly_times = [
        forecast_start + timedelta(hours=hour)
        for hour in range(hours_ahead)
    ]

    df_forecast = pd.DataFrame({"Time": hourly_times})

    if "temperature" in forecast_results:
        temperatures = forecast_results["temperature"]

        if temperature_unit == "F":
            temperatures = [
                (temperature * 9 / 5) + 32
                for temperature in temperatures
            ]

        displayed_temp_unit = (
            "°م" if is_ar and temperature_unit == "C"
            else "°ف" if is_ar
            else "°C" if temperature_unit == "C"
            else "°F"
        )

        temperature_label = (
            "درجة الحرارة"
            if is_ar
            else "Temperature"
        )

        df_forecast[
            f"{temperature_label} ({displayed_temp_unit})"
        ] = temperatures

    if "humidity" in forecast_results:
        humidity_label = "الرطوبة" if is_ar else "Humidity"
        df_forecast[
            f"{humidity_label} (%)"
        ] = forecast_results["humidity"]

    if "wind_speed" in forecast_results:
        wind_values = forecast_results["wind_speed"]

        if wind_unit == "ms":
            wind_values = [
                value / 3.6
                for value in wind_values
            ]

        displayed_wind_unit = (
            "كم/سا" if is_ar and wind_unit == "kmh"
            else "م/ث" if is_ar
            else "km/h" if wind_unit == "kmh"
            else "m/s"
        )

        wind_label = "سرعة الرياح" if is_ar else "Wind Speed"

        df_forecast[
            f"{wind_label} ({displayed_wind_unit})"
        ] = wind_values

    # ---------- Plot results ----------
    def plot_line_chart(
        forecast_df: pd.DataFrame,
        column: str,
        title: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            forecast_df["Time"],
            forecast_df[column],
            marker="o",
        )

        ax.set_title(title)
        ax.set_xlabel("الوقت" if is_ar else "Time")
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader(
        "توقعات الطقس لكل ساعة غداً"
        if is_ar
        else "Hourly Weather Forecast for Tomorrow"
    )

    st.markdown(f"**{city}, {country}**")
    st.markdown(f"**{forecast_day.isoformat()}**")

    for column in df_forecast.columns:
        if column == "Time":
            continue

        label = column.split(" (")[0]

        chart_title = (
            f"{label} خلال اليوم"
            if is_ar
            else f"{label} Throughout the Day"
        )

        plot_line_chart(
            df_forecast,
            column,
            chart_title,
        )

    st.dataframe(
        df_forecast.style.format(precision=1),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "بيانات الطقس مقدمة من Open-Meteo"
        if is_ar
        else "Weather data provided by Open-Meteo"
    )
