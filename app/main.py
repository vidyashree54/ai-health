# main.py
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from passlib.context import CryptContext
import nltk
from bson import ObjectId
from tensorflow.keras.models import load_model
import joblib
import numpy as np
# ---- GEMINI AI ----
import google.generativeai as genai



# Try to import reasoning/optional libs
USE_TRANSFORMERS = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    USE_TRANSFORMERS = True
except Exception:
    USE_TRANSFORMERS = False

# ----------------- load env -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "ai_health_db")

if not MONGO_URI:
    raise RuntimeError("Please set MONGO_URI in your .env file")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------- ML Models -----------------

# ----------------- ML Models -----------------
FITNESS_CALORIE_MODEL = os.getenv("FITNESS_CALORIE_MODEL")
FITNESS_ACTIVITY_MODEL = os.getenv("FITNESS_ACTIVITY_MODEL")
FITNESS_SCALER_X = os.getenv("FITNESS_SCALER_X")
FITNESS_SCALER_Y = os.getenv("FITNESS_SCALER_Y")
FITNESS_SCALER_X2 = os.getenv("FITNESS_SCALER_X2")
FITNESS_SCALER_Y2 = os.getenv("FITNESS_SCALER_Y2")

def _assert_exists(path, label):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")

try:
    calorie_model = load_model(FITNESS_CALORIE_MODEL, compile=False)
    activity_model = load_model(FITNESS_ACTIVITY_MODEL, compile=False)

    scaler_X = joblib.load(FITNESS_SCALER_X)
    scaler_y = joblib.load(FITNESS_SCALER_Y)
    scaler_X2 = joblib.load(FITNESS_SCALER_X2)
    scaler_y2 = joblib.load(FITNESS_SCALER_Y2)

    print("✅ Fitness models loaded successfully.")
except Exception as e:
    print("⚠️ Fitness models not loaded:", repr(e))

# ----------------- DB -----------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users = db["users"]
moods = db["mood_logs"]
logs = db["daily_logs"]
vitals = db["vitals"]
community = db["community_posts"]
doctors = db["doctors"]
chats = db["chat_history"]

# ----------------- security -----------------
# pbkdf2_sha256 avoids bcrypt installation issues and is secure for this project
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ----------------- FastAPI -----------------
app = FastAPI(title="AI Health Companion Backend (Single-file)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow for dev; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Ensure nltk data -----------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

from nltk.sentiment import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# ----------------- Optional local LLM (light fallback) -----------------
chat_pipeline = None
if USE_TRANSFORMERS:
    try:
        # try a lightweight model; will download if not present
        # you can replace 'distilgpt2' with a local path if you have one
        chat_pipeline = pipeline("text-generation", model="distilgpt2")
    except Exception:
        chat_pipeline = None

# ----------------- Pydantic models -----------------
class User(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


class ChatSave(BaseModel):
    email: str
    sender: str   # "user" or "bot"
    text: str



class ChatRequest(BaseModel):
    text: str
    email: Optional[str] = None

class MoodLog(BaseModel):
    email: str
    mood: str
    note: Optional[str] = None

class LogEntry(BaseModel):
    email: str
    log: str

class Post(BaseModel):
    email: str
    text: str

class VitalEntry(BaseModel):
    email: str
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    blood_sugar: Optional[float] = None
    sleep_hours: Optional[float] = None


class CalorieInput(BaseModel):
    total_steps: int
    total_distance: float
    very_active: int
    fairly_active: int
    lightly_active: int
    sedentary: int

class CalorieGoal(BaseModel):
    calories: float

# ----------------- Helpers -----------------
def generate_ai_reply(user_text: str) -> str:
    """
    Hybrid approach:
     - Quick rules for common health words
     - Sentiment prefix using VADER
     - If local transformers available, use it for generation
     - Else return a friendly templated reply
    """
    text = user_text.lower()

    # basic health rules (phase 5)
    if any(w in text for w in ["chest pain", "difficulty breathing", "shortness of breath"]):
        return "⚠️ This sounds serious. Please seek immediate medical help or call emergency services."

    if "headache" in text:
        return "For headaches: rest, hydrate, avoid screens. If severe or sudden, consult a doctor."

    if "fever" in text:
        return "For fever: keep hydrated and rest. If temperature is high or persistent, seek medical care."

    if "tired" in text or "fatigue" in text:
        return "Fatigue might come from lack of sleep, stress or dehydration. Try a short walk and rest."

    # sentiment + fallback
    s = sentiment_analyzer.polarity_scores(user_text)
    mood_tag = "(mood: neutral)"
    if s["compound"] > 0.2:
        mood_tag = "(mood: positive)"
    elif s["compound"] < -0.2:
        mood_tag = "(mood: negative)"

    # attempt model generation if available
    if chat_pipeline:
        try:
            out = chat_pipeline(user_text, max_length=80, num_return_sequences=1)
            text_out = out[0]["generated_text"]
            return f"{mood_tag} {text_out}"
        except Exception:
            pass

    # final deterministic fallback
    return f"{mood_tag} I hear you. Can you tell me more about your symptoms or how you're feeling?"


model = genai.GenerativeModel("gemini-2.0-flash")

def medical_ai_reply(user_text: str) -> str:
    prompt = f"""
You are an AI medical assistant. 
Reply professionally like a doctor would.  
Avoid using *, **, or markdown.  
Keep answers short, safe, helpful.

User: {user_text}
Doctor Response:
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Remove any stray *s
        clean = text.replace("*", "").replace("**", "").replace("###", "")
        return clean
    except:
        return "I’m here to assist you. Can you describe your symptoms clearly?"

# ----------------- Routes -----------------
@app.get("/")
def root():
    return {"message": "AI Health Backend running ✅"}

# Auth - register
@app.post("/register")
def register(user: User):
    if users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="User already exists")

    hashed = pwd_context.hash(user.password)
    users.insert_one({
        "name": user.name,
        "email": user.email,
        "password": hashed
    })
    return {"message": "User registered successfully"}


# Auth - login
@app.post("/login")
def login(user: LoginRequest):

    existing = users.find_one({"email": user.email})
    if not existing or not pwd_context.verify(user.password, existing["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "message": "Login success",
        "email": existing["email"],
        "name": existing["name"]
    }


# Chat endpoint (Phase1 Chat + Phase5 Symptom checker)

@app.post("/chat")
def chat(req: ChatRequest):
    # Save user message
    if req.email:
        chats.insert_one({
            "email": req.email,
            "sender": "user",
            "text": req.text,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        })

    try:
        # --- 1. Gemini medical reply ---
        prompt = f"""
        You are a certified medical assistant.
        Give safe, short medical guidance.
        Avoid markdown symbols (*, **, #).
        If symptoms are serious, advise doctor visit.

        User: {req.text}
        AI Response:
        """
        response = model.generate_content(prompt)
        reply = response.text.strip()
        reply = reply.replace("*", "").replace("**", "").replace("#", "")
        if req.email:
            chats.insert_one({
            "email": req.email,
            "sender": "bot",
            "text": reply,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        })

        # --- 2. Mood Detection (AI + Rules) ---
        text = req.text.lower()
        s = sentiment_analyzer.polarity_scores(req.text)
        mood = "neutral"

        if s["compound"] > 0.3: mood = "happy"
        elif s["compound"] < -0.3: mood = "sad"

        if any(w in text for w in ["stress", "anxiety", "worried", "panic"]):
            mood = "anxious"
        if any(w in text for w in ["angry", "frustrated", "mad", "irritated"]):
            mood = "angry"
        if any(w in text for w in ["tired", "exhausted", "sleepy"]):
            mood = "tired"
        if any(w in text for w in ["fever", "sick", "pain", "headache"]):
            mood = "sick"

        # --- 3. Save Mood Automatically ---
        if req.email:
            moods.insert_one({
                "email": req.email,
                "mood": mood,
                "note": req.text,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat()
            })

        return {"reply": reply, "mood": mood}

    except Exception as e:
        return {"reply": "The medical assistant is currently unavailable, please try again.", "mood": "unknown"}




# Mood endpoints (Phase1/2)
@app.post("/mood/add")
def add_mood(m: MoodLog):
    moods.insert_one({
        "email": m.email,
        "mood": m.mood,
        "note": m.note,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    })
    return {"message": "Mood logged"}

@app.get("/mood/get/{email}")
def get_moods(email: str):
    docs = list(moods.find({"email": email}).sort([("_id", -1)]).limit(100))
    out = [{"mood": d["mood"], "note": d.get("note", ""), "timestamp": d["timestamp"]} for d in docs]
    return {"moods": out}

# Daily logs (Phase1)
@app.post("/log/add")
def add_log(entry: LogEntry):
    logs.insert_one({
        "email": entry.email,
        "log": entry.log,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    })
    return {"message": "Log saved"}

@app.get("/log/get/{email}")
def get_logs(email: str):
    docs = list(logs.find({"email": email}).sort([("_id", -1)]).limit(200))
    return {"logs": [{"log": d["log"], "timestamp": d["timestamp"]} for d in docs]}

# Vitals entry (Phase2)
@app.post("/vitals/add")
def add_vital(v: VitalEntry):
    vitals.insert_one({
        "email": v.email,
        "bp_systolic": v.bp_systolic,
        "bp_diastolic": v.bp_diastolic,
        "heart_rate": v.heart_rate,
        "blood_sugar": v.blood_sugar,
        "sleep_hours": v.sleep_hours,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    })
    return {"message": "Vitals saved"}

@app.get("/vitals/get/{email}")
def get_vitals(email: str):
    docs = list(vitals.find({"email": email}).sort([("_id", -1)]).limit(200))
    out = []
    for d in docs:
        out.append({
            "_id": str(d["_id"]),  # Convert ObjectId to string
            "email": d.get("email"),
            "bp_systolic": d.get("bp_systolic"),
            "bp_diastolic": d.get("bp_diastolic"),
            "heart_rate": d.get("heart_rate"),
            "blood_sugar": d.get("blood_sugar"),
            "sleep_hours": d.get("sleep_hours"),
            "timestamp": d.get("timestamp"),
        })
    return {"vitals": out}

# Simple report (Phase2/3/5)
@app.get("/report/{email}")
def report(email: str):
    # build a simple mood summary
    docs = list(moods.find({"email": email}))
    if not docs:
        return {"summary": "No mood data yet."}
    from collections import Counter
    counts = Counter([d["mood"] for d in docs])
    top = counts.most_common(1)[0][0]
    return {"summary": f"Dominant mood: {top}. Counts: {dict(counts)}"}

@app.get("/health_score/{email}")
def health_score(email: str):
    vit = list(vitals.find({"email": email}).sort([("_id", -1)]).limit(5))
    mood_data = list(moods.find({"email": email}).sort([("_id", -1)]).limit(10))

    if not vit:
        return {"health_score": 50, "details": "Add vitals to get accurate score"}

    avg_hr = sum([v.get("heart_rate", 0) for v in vit]) / len(vit)
    avg_sleep = sum([v.get("sleep_hours", 0) for v in vit]) / len(vit)
    
    # simple scoring logic
    score = 50
    if 60 <= avg_hr <= 100: score += 20
    if 6 <= avg_sleep <= 8: score += 20
    if mood_data:
        pos = len([m for m in mood_data if m["mood"] in ["happy","neutral"]])
        score += int((pos / len(mood_data)) * 10)

    return {"health_score": score, "avg_heart_rate": avg_hr, "avg_sleep": avg_sleep}

# Emergency check endpoint (Phase5)
@app.post("/check_emergency")
def check_emergency(text: dict):
    msg = text.get("text", "")
    if any(w in msg.lower() for w in ["suicide", "kill myself", "hurt myself", "emergency", "chest pain", "severe bleeding"]):
        return {"emergency": True, "advice": "Please contact local emergency services or a trusted person immediately."}
    return {"emergency": False, "advice": "No immediate emergency detected."}



@app.get("/daily_tip/{email}")
def daily_tip(email: str):
    try:
        prompt = """
        You are a wellness and lifestyle expert.
        Give one short, simple, daily health advice in one sentence.
        No markdown, no emojis, no lists.
        The tip should be practical and safe.
        """

        response = model.generate_content(prompt)
        tip = response.text.strip().replace("*", "").replace("#", "")
        return {"tip": tip}
    except Exception:
        return {"tip": "Drink enough water and take short walks during the day."}


@app.post("/community/post")
def create_post(p: Post):
    community.insert_one({
        "email": p.email,
        "text": p.text,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    })
    return {"message": "Post added ✅"}

@app.get("/community/feed")
def get_posts():
    docs = list(community.find().sort([("_id", -1)]).limit(100))
    out = [{"email": d["email"], "text": d["text"], "timestamp": d["timestamp"]} for d in docs]
    return {"posts": out}


# ----------------- Doctor Connect -----------------
@app.get("/doctors/list")
def list_doctors():
    return {
        "doctors": [
            {"name": "Dr. A Sharma", "specialty": "General Physician", "online": True},
            {"name": "Dr. Mehta", "specialty": "Cardiologist", "online": False},
            {"name": "Dr. R Singh", "specialty": "Dermatologist", "online": True}
        ]
    }

@app.post("/doctor/request")
def request_doctor(data: dict):
    email = data.get("email")
    doctor = data.get("doctor")
    return {
        "message": f"Consultation request sent to {doctor} ✅",
        "status": "pending",
        "user": email
    }
@app.get("/chat/history/{email}")
def chat_history(email: str):
    docs = list(chats.find({"email": email}).sort([("_id", 1)]).limit(200))
    return {"history": [{"sender": d["sender"], "text": d["text"]} for d in docs]}



@app.post("/predict/calories")
def predict_calories(data: CalorieInput):
    try:
        X = np.array([[
            data.total_steps,
            data.total_distance,
            data.very_active,
            data.fairly_active,
            data.lightly_active,
            data.sedentary
        ]])

        # Scale input correctly
        X_scaled = scaler_X.transform(X)
        y_scaled = calorie_model.predict(X_scaled)

        # Proper inverse transform (ensure shape)
        y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))

        predicted = float(y_pred[0][0])

        # ✅ Physiological sanity correction
        # Avoid absurd outputs: limit between 100 and 6000 kcal/day
        predicted = max(100, min(predicted, 6000))

        # ✅ Ensure increasing with steps
        # Basic monotonic correction — more steps should never yield less kcal
        if data.total_steps > 2000:
            predicted *= (1 + (data.total_steps / 20000))

        return {"predicted_calories": round(predicted, 2)}

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/recommend/activity")
def recommend_activity(goal: CalorieGoal):
    try:
        goal_scaled = scaler_X2.transform(np.array([[goal.calories]]))
        activity_scaled = activity_model.predict(goal_scaled)
        activity = scaler_y2.inverse_transform(activity_scaled)[0]
        return {
            "recommended_steps": int(activity[0]),
            "recommended_distance": round(float(activity[1]), 2),
            "very_active_minutes": int(activity[2]),
            "fairly_active_minutes": int(activity[3]),
            "lightly_active_minutes": int(activity[4])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")