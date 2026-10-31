"""
AI Health & Air Quality Advisor Module for Rooted AirSense
Grounded LLM Advisor powered by Google Gemini Flash with Groq fallback.
Answers public health, sensitive group, and neighborhood-specific air quality questions.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Try importing Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Try importing OpenAI client for Groq fallback
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


def resolve_gemini_api_key() -> Tuple[Optional[str], str]:
    """
    Resolves the Gemini API Key with precedence:
    1. Local .env file (via python-dotenv): GEMINI_API_KEY
    2. Local .env fallback: GEMINI_API_KEY
    3. Streamlit Cloud secrets: st.secrets["GEMINI_API_KEY"]
    4. Streamlit Cloud secrets fallback: st.secrets["GEMINI_API_KEY"] or st.secrets["GOOGLE_API_KEY"]
    """
    env_gemini = os.getenv("GEMINI_API_KEY")
    if env_gemini and env_gemini.strip():
        return env_gemini.strip(), "Local .env (GEMINI_API_KEY)"

    env_token = os.getenv("GEMINI_API_KEY")
    if env_token and env_token.strip():
        return env_token.strip(), "Local .env (GEMINI_API_KEY...)"

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                val = st.secrets["GEMINI_API_KEY"]
                if val and str(val).strip():
                    return str(val).strip(), "st.secrets[GEMINI_API_KEY...]"
            if "GEMINI_API_KEY" in st.secrets:
                val = st.secrets["GEMINI_API_KEY"]
                if val and str(val).strip():
                    return str(val).strip(), "st.secrets[GEMINI_API_KEY]"
            if "GOOGLE_API_KEY" in st.secrets:
                val = st.secrets["GOOGLE_API_KEY"]
                if val and str(val).strip():
                    return str(val).strip(), "st.secrets[GOOGLE_API_KEY]"
    except Exception:
        pass

    return None, "Not found"


def resolve_groq_api_key() -> Tuple[Optional[str], str]:
    """
    Resolves the Groq API Key from local .env or st.secrets.
    """
    env_groq = os.getenv("GROQ_API_KEY")
    if env_groq and env_groq.strip():
        return env_groq.strip(), "Local .env (GROQ_API_KEY)"

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            val = st.secrets["GROQ_API_KEY"]
            if val and str(val).strip():
                return str(val).strip(), "st.secrets[GROQ_API_KEY]"
    except Exception:
        pass

    return None, "Not found"


def build_airsense_system_prompt(
    neighborhood_predictions: Optional[List[Dict[str, Any]]] = None,
    weather_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Builds a dynamically grounded system prompt containing real-time Patiala
    neighborhood AQI levels, stubble smoke status, and clinical health guidelines.
    """
    context_lines = [
        "You are 'Rooted AirSense 🌱', an expert environmental health AI and neighborhood air quality advisor for Patiala, Punjab.",
        "Your mission is to provide actionable, empathetic, scientifically accurate, and locally relevant guidance to citizens, schools, asthmatics, and families.",
        "",
        "CURRENT ENVIRONMENTAL CONTEXT IN PATIALA:",
    ]

    if weather_snapshot:
        wind_dir = weather_snapshot.get("wind_direction", 270)
        wind_spd = weather_snapshot.get("wind_speed", 8)
        temp = weather_snapshot.get("temperature", 22)
        hum = weather_snapshot.get("humidity", 60)
        context_lines.append(
            f"- Ambient Weather: Temp {temp}°C, Humidity {hum}%, Wind {wind_spd} km/h from {wind_dir}°."
        )

    if neighborhood_predictions:
        context_lines.append("- Predicted Neighborhood Pollution Levels:")
        for pred in neighborhood_predictions[:8]:
            context_lines.append(
                f"  * {pred['name']}: AQI {pred['predicted_aqi']} ({pred['category']}) | "
                f"PM2.5: {pred['predicted_pm25']} µg/m³ | Zone: {pred.get('zone_type', 'Mixed')}"
            )

    context_lines.extend([
        "",
        "KEY ADVISORY INSTRUCTIONS:",
        "1. Give concrete neighborhood-specific advice (e.g. mention Baradari Gardens as a cleaner pocket vs Focal Point or Leela Bhawan as smog hotspots).",
        "2. Address vulnerable populations (children, asthmatics, elderly, pregnant women) with specific protections like N95 masks, air purifiers, and avoiding outdoor runs during evening inversion hours.",
        "3. Explain the influence of stubble burning from western Malwa districts and wind transport clearly.",
        "4. Keep your tone encouraging, proactive, eco-conscious, and clear.",
    ])

    return "\n".join(context_lines)


def build_pehredar_system_prompt(
    locality_summary: Optional[List[Dict[str, Any]]] = None,
    weather_summary: Optional[Dict[str, Any]] = None,
    metrics_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Builds system prompt for Pehredar AI advisor grounded on H3 neighborhood forecasts,
    LightGBM predictions, Safety Indices, and Malwa farm belt wind vectors.
    """
    context_lines = [
        "You are 'Pehredar 🛡️', an AI-powered neighborhood environmental intelligence advisor for Patiala, Punjab.",
        "Your mission is to provide clear, actionable, and compassionate guidance to citizens regarding air quality, health precautions, and local exposure.",
        "",
        "CURRENT PREDICTION CONTEXT FOR PATIALA:",
    ]
    if weather_summary:
        context_lines.append(
            f"- Ambient Conditions: Wind {weather_summary.get('wind_speed_10m', 'N/A')} km/h from {weather_summary.get('wind_direction_10m', 'N/A')}°, "
            f"Stubble belt transport: {'Active' if weather_summary.get('is_stubble_wind') else 'Low'}."
        )
    if locality_summary:
        context_lines.append("- Locality Forecast Snapshot:")
        for loc in locality_summary[:10]:
            context_lines.append(
                f"  * {loc['locality']}: Expected PM2.5 {loc['pm2_5']} µg/m³ (AQI {loc['aqi']}, {loc['aqi_category']}) | "
                f"Safety Index: Healthy {loc.get('safety_healthy', 'N/A')}/10, Asthma {loc.get('safety_respiratory', 'N/A')}/10"
            )
    if metrics_info:
        context_lines.append(
            f"- Prediction Engine: LightGBM Multi-Horizon Quantile Regressor (MAE: {metrics_info.get('model_mae')} µg/m³ vs Baseline: {metrics_info.get('baseline_mae')} µg/m³)."
        )

    context_lines.extend([
        "",
        "KEY ADVISORY INSTRUCTIONS:",
        "1. Give direct answers with specific locality names (e.g. Baradari, Leela Bhawan, Model Town, Thapar University, Tripuri, Urban Estate).",
        "2. State the Safety Index (1 to 10) for healthy adults, children, and people with asthma or lung conditions.",
        "3. Recommend practical protective measures: optimal ventilation times, N95 masks during high-risk hours, indoor purifiers, and avoiding outdoor runs during evening peak smog.",
        "4. Keep answers concise, factual, and reassuring.",
    ])
    return "\n".join(context_lines)


def generate_response_with_resilience(
    selected_gemini_model: str,
    gemini_key: Optional[str],
    groq_api_key: Optional[str],
    formatted_gemini_contents: list,
    openai_messages: list,
    temp: float,
    sys_instruction: str,
    spinner_container: Any = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Executes resilient inference with automatic retries (2s, then 4s wait on temporary errors)
    and graceful cascade fallback:
      gemini-3.8-flash → gemini-3.6-flash → gemini-3.5-flash-lite → Groq (openai/gpt-oss-120b)
    """
    retry_delays = [2, 4]

    # Build Gemini model cascade based on user's starting choice
    cascade_order = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    if selected_gemini_model in cascade_order:
        start_idx = cascade_order.index(selected_gemini_model)
        gemini_models_to_try = cascade_order[start_idx:]
    else:
        gemini_models_to_try = [selected_gemini_model] + [
            m for m in cascade_order if m != selected_gemini_model
        ]

    # Helper for retry wait with spinner
    def wait_with_spinner(seconds: int):
        try:
            import streamlit as st
            with st.spinner("Thinking a bit longer..."):
                time.sleep(seconds)
        except Exception:
            time.sleep(seconds)

    # 1. Attempt Gemini Models with retries
    if gemini_key and genai is not None and types is not None:
        try:
            client = genai.Client(api_key=gemini_key)
            config = types.GenerateContentConfig(
                temperature=temp,
                system_instruction=sys_instruction.strip() if sys_instruction else None,
            )

            for model_name in gemini_models_to_try:
                for attempt in range(len(retry_delays) + 1):
                    try:
                        response_stream = client.models.generate_content_stream(
                            model=model_name,
                            contents=formatted_gemini_contents,
                            config=config,
                        )
                        stream_iter = iter(response_stream)
                        # Probe the first chunk to catch immediate 429/503/auth errors
                        first_chunk = next(stream_iter)

                        def gemini_stream_generator(first, rest):
                            try:
                                if first and first.text:
                                    yield first.text
                                for chunk in rest:
                                    if chunk and chunk.text:
                                        yield chunk.text
                            except Exception:
                                pass

                        return gemini_stream_generator(first_chunk, stream_iter), model_name

                    except Exception:
                        if attempt < len(retry_delays):
                            wait_with_spinner(retry_delays[attempt])
                        else:
                            break
        except Exception:
            pass

    # 2. Attempt Groq fallback via OpenAI SDK if GROQ_API_KEY is available
    if groq_api_key and OpenAI is not None:
        try:
            groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_api_key,
            )
            for attempt in range(len(retry_delays) + 1):
                try:
                    groq_stream = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=openai_messages,
                        temperature=temp,
                        stream=True,
                    )
                    stream_iter = iter(groq_stream)
                    first_chunk = next(stream_iter)

                    def groq_stream_generator(first, rest):
                        try:
                            if first.choices and first.choices[0].delta.content:
                                yield first.choices[0].delta.content
                            for chunk in rest:
                                if chunk.choices and chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content
                        except Exception:
                            pass

                    return groq_stream_generator(first_chunk, stream_iter), "openai/gpt-oss-120b (via Groq)"

                except Exception:
                    if attempt < len(retry_delays):
                        wait_with_spinner(retry_delays[attempt])
                    else:
                        break
        except Exception:
            pass

    return None, None
