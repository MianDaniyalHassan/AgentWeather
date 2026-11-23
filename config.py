import os
from dotenv import load_dotenv

load_dotenv()

CITY = "Lahore"
LAT = 31.5204
LON = 74.3587
TIMEZONE = "Asia/Karachi"

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
AQICN_STATION = os.getenv("AQICN_STATION")

LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
