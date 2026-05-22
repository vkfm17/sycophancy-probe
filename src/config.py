import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PROBE_MODEL = os.getenv("PROBE_MODEL", "claude-opus-4-6")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")

RESULTS_DIR = "results"
DATA_DIR = "data"
