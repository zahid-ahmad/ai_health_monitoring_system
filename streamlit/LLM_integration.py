import streamlit as st
import pandas as pd
import os
from datetime import datetime
import requests

# Try to import the OpenAI client wrapper for the HF router (optional)
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False

# -------------------------------
# Hugging Face / Router setup
# -------------------------------
# Support both env names HF_TOKEN and HUGGINGFACE_API_TOKEN
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
# Candidate models to try (adjust/add hosted HF model IDs you want)
MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "gpt2",
    "distilgpt2",
    "microsoft/DialoGPT-medium",
    "meta-llama/Llama-3.1-8B-Instruct"
]

def _call_router_with_openai_client(model: str, prompt: str, timeout: int = 30):
    """Use the OpenAI-compatible client pointed at HF router."""
    if not HF_API_TOKEN:
        raise RuntimeError("HUGGINGFACE API token not set (HF_TOKEN or HUGGINGFACE_API_TOKEN).")
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_TOKEN)
    # Use chat completions interface (many HF models accept user messages)
    messages = [
        {"role": "system", "content": "You are a concise, helpful health assistant."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=256, temperature=0.2)

    # Extract content robustly from many possible response shapes
    text = None
    try:
        choice = resp.choices[0]
        # possibility: choice is an object with .message
        if hasattr(choice, "message") and choice.message is not None:
            msg = choice.message
            # message may be object with .content or a dict
            if hasattr(msg, "content"):
                text = msg.content
            elif isinstance(msg, dict):
                text = msg.get("content") or msg.get("text")
        elif isinstance(choice, dict):
            # dict shape: {'message': {'content': '...'}} or {'text': '...'}
            msg = choice.get("message")
            if isinstance(msg, dict):
                text = msg.get("content") or msg.get("text")
            text = text or choice.get("text") or choice.get("message") or None

        # final fallbacks
        if not text:
            # try common attribute names or string-repr
            text = getattr(choice, "text", None) or str(choice)
    except Exception:
        text = str(resp)

    return {"model": model, "text": text}

def _call_inference_api_requests(model: str, prompt: str, timeout: int = 30):
    """Direct call to HF Inference API (api-inference.huggingface.co)."""
    if not HF_API_TOKEN:
        raise RuntimeError("HUGGINGFACE API token not set (HF_TOKEN or HUGGINGFACE_API_TOKEN).")
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.2}}
    url = f"https://api-inference.huggingface.co/models/{model}"
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 404:
        raise FileNotFoundError(f"Model not found for inference: {model}")
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return {"model": model, "text": data[0]["generated_text"]}
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Inference API error from {model}: {data['error']}")
    return {"model": model, "text": str(data)}

def query_model_via_router(model: str, prompt: str, timeout: int = 30):
    """Try OpenAI client (router) first if available, else fall back to requests router path."""
    last_exc = None
    if OPENAI_CLIENT_AVAILABLE:
        try:
            return _call_router_with_openai_client(model, prompt, timeout=timeout)
        except Exception as e:
            last_exc = e
            # continue to try requests-based inference endpoint below
    try:
        # Some HF setups accept the router v1 endpoint via requests too (older path used earlier).
        # Primary requests fallback uses api-inference.huggingface.co
        return _call_inference_api_requests(model, prompt, timeout=timeout)
    except Exception as e:
        # if openai client failed earlier, prefer that error for debugging
        if last_exc:
            raise RuntimeError(f"OpenAI client error: {last_exc}\nRequests fallback error: {e}")
        raise

def query_best_model(prompt: str, timeout: int = 30, candidates=None):
    """Try candidate models (router/OpenAI client -> inference API) until one returns a response."""
    candidates = candidates or MODEL_CANDIDATES
    errors = {}
    for m in candidates:
        try:
            return query_model_via_router(m, prompt, timeout=timeout)
        except FileNotFoundError as fnf:
            errors[m] = str(fnf)
            continue
        except Exception as e:
            errors[m] = str(e)
            continue
    details = "\n".join(f"{mod}: {err}" for mod, err in errors.items())
    raise RuntimeError(f"No available models succeeded. Attempts:\n{details}")

# -------------------------------
# Data Storage
# -------------------------------
DATA_FILE = "health_data.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["DateTime", "Heart Rate", "BP Systolic", "BP Diastolic", "Oxygen", "Temperature"])

# -------------------------------
# App Layout
# -------------------------------
st.title("🏥 AI-Powered Virtual Health Assistant")

tabs = st.tabs(["📊 Health Log", "📈 Trends", "🤖 Chatbot Assistant"])

# -------------------------------
# Tab 1: Health Log
# -------------------------------
with tabs[0]:
    with st.form("health_form"):
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=30, max_value=200, value=72)
            bp_systolic = st.number_input("🩸 BP - Systolic", min_value=80, max_value=200, value=120)
        with col2:
            bp_diastolic = st.number_input("🩸 BP - Diastolic", min_value=40, max_value=130, value=80)
            oxygen = st.number_input("🫁 Oxygen (%)", min_value=50, max_value=100, value=98)
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=30.0, max_value=43.0, value=36.8, step=0.1)
        
        submitted = st.form_submit_button("Submit Vitals")

    if submitted:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = {"DateTime": now, "Heart Rate": heart_rate, "BP Systolic": bp_systolic,
                    "BP Diastolic": bp_diastolic, "Oxygen": oxygen, "Temperature": temperature}
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("✅ Data submitted successfully!")

    st.subheader("📊 Your Health Log")
    st.dataframe(df)

# -------------------------------
# Tab 2: Trends
# -------------------------------
with tabs[1]:
    st.subheader("📈 Trends Over Time")
    if not df.empty:
        st.line_chart(df.set_index("DateTime")[["Heart Rate", "BP Systolic", "BP Diastolic", "Oxygen", "Temperature"]])
    else:
        st.info("No data available yet.")

# -------------------------------
# Tab 3: Chatbot
# -------------------------------
with tabs[2]:
    st.subheader("🤖 Health Assistant Chatbot")

    if not HF_API_TOKEN:
        st.warning("Hugging Face API token not set. To enable the chatbot set HUGGINGFACE_API_TOKEN or HF_TOKEN in the environment.")
        st.write("How to set token in this devcontainer (bash):")
        st.code('export HUGGINGFACE_API_TOKEN="hf_xxx"')

    chosen_model = st.selectbox("Choose HF model (attempted in order):", options=MODEL_CANDIDATES, index=0)

    user_q = st.text_input("Ask me anything about health (e.g., What does high BP mean?)")

    if user_q:
        # Build BP-focused context from stored data
        bp_entries = []
        if not df.empty and "BP Systolic" in df.columns and "BP Diastolic" in df.columns:
            # collect last up to 6 BP measurements as tuples (systolic, diastolic, datetime)
            for _, row in df.tail(6).iterrows():
                try:
                    s = int(row.get("BP Systolic"))
                    d = int(row.get("BP Diastolic"))
                except Exception:
                    # if stored as "120/80" try parse
                    bp_raw = row.get("Blood Pressure") or ""
                    if isinstance(bp_raw, str) and "/" in bp_raw:
                        parts = bp_raw.split("/")
                        try:
                            s = int(parts[0].strip()); d = int(parts[1].strip())
                        except Exception:
                            continue
                    else:
                        continue
                bp_entries.append({"systolic": s, "diastolic": d, "datetime": row.get("DateTime") or row.get("Date")})

        # current and previous formatting
        if bp_entries:
            current = bp_entries[-1]
            prev_list = bp_entries[:-1] or []
            prev_str = ", ".join(f"{e['systolic']}/{e['diastolic']}" for e in prev_list) if prev_list else "none"
            current_str = f"{current['systolic']}/{current['diastolic']}"
            # simple trend: compare mean of last 3 vs previous 3 if possible
            def mean_bp(entries):
                if not entries:
                    return None
                s_mean = sum(e["systolic"] for e in entries)/len(entries)
                d_mean = sum(e["diastolic"] for e in entries)/len(entries)
                return (s_mean, d_mean)
            last3 = bp_entries[-3:] if len(bp_entries) >= 3 else bp_entries
            prev3 = bp_entries[-6:-3] if len(bp_entries) >= 6 else (bp_entries[:-3] if len(bp_entries) > 3 else [])
            m_last = mean_bp(last3); m_prev = mean_bp(prev3)
            trend = "stable"
            if m_last and m_prev:
                # compare systolic primarily
                if m_last[0] >= m_prev[0] + 5:
                    trend = "increasing"
                elif m_last[0] <= m_prev[0] - 5:
                    trend = "decreasing"
                else:
                    trend = "stable"
        else:
            current_str = "no_data"
            prev_str = "no_data"
            trend = "no_data"

        # collect latest symptoms if available in df
        latest_symptoms = ""
        if "Symptoms" in df.columns and not df.empty:
            latest_symptoms = df.tail(1).iloc[0].get("Symptoms", "") or ""
        # Compose strict instruction prompt: only output required fields and nothing else.
        prompt = (
            f"Intract politely to the user.\n"
            f"You are a concise assistant. Use only the BP values and trends below to answer the user's question.\n\n"
            f"Current_BP: {current_str}\n"
            f"Previous_BP: {prev_str}\n"
            f"Trend: {trend}\n"
            f"Noted_Symptoms: {latest_symptoms or 'none'}\n\n"
            f"User question: {user_q}\n\n"
            "Output using following items:\n"
            "Current_BP: <systolic/diastolic>\n"
            "Previous_BP: <comma-separated list or 'none'>\n"
            "Trend: <increasing|decreasing|stable|no_data>\n"
            "Suggestions: <short bullet points or comma-separated suggestions>\n"
            "Noted_Symptoms: <copy from data or 'none'>\n\n"
        )

        try:
            # try router client first then fallbacks
            try:
                result = query_model_via_router(chosen_model, prompt)
            except Exception:
                candidate_list = [chosen_model] + [c for c in MODEL_CANDIDATES if c != chosen_model]
                result = query_best_model(prompt, candidates=candidate_list)
            # display the raw assistant output (should conform to strict format)
            st.write(result["text"].strip())
        except Exception as e:
            st.error("Chatbot error: unable to get a reply from hosted models.")
            st.write(f"Details: {e}")
            if not HF_API_TOKEN:
                st.info('Set HUGGINGFACE_API_TOKEN or HF_TOKEN in the environment (e.g. export HUGGINGFACE_API_TOKEN="hf_xxx").')
            else:
                st.info(f"Models attempted: {', '.join([chosen_model] + [c for c in MODEL_CANDIDATES if c != chosen_model])}")
            # deterministic fallback: produce a minimal structured reply from local data
            fallback_suggestions = []
            if current_str != "no_data":
                s_val = int(current_str.split("/")[0])
                if s_val >= 140:
                    fallback_suggestions.append("seek medical review")
                elif s_val >= 130:
                    fallback_suggestions.append("monitor closely, consider lifestyle changes")
                else:
                    fallback_suggestions.append("maintain healthy habits")
            st.write("Current_BP:", current_str)
            st.write("Previous_BP:", prev_str)
            st.write("Trend:", trend)
            st.write("Suggestions:", ", ".join(fallback_suggestions) or "none")
            st.write("Noted_Symptoms:", latest_symptoms or "none")
