"""Central configuration: paths, vocabulary and real-time tuning values."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent

ASSETS_DIR = BACKEND_DIR / "assets"
DATA_DIR = BACKEND_DIR / "data"
OWN_DIR = DATA_DIR / "own"                 # recordings made with the app's Record page
PROCESSED_DIR = DATA_DIR / "processed"     # public datasets converted to landmarks
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"
DB_PATH = DATA_DIR / "signvision.db"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

HAND_MODEL = ASSETS_DIR / "hand_landmarker.task"
STATIC_MODEL = MODELS_DIR / "static_model.pt"
DYNAMIC_MODEL = MODELS_DIR / "dynamic_model.pt"
SENTENCE_MODEL = MODELS_DIR / "sentence_model.pt"
MODEL_PATHS = {"static": STATIC_MODEL, "dynamic": DYNAMIC_MODEL, "sentence": SENTENCE_MODEL}
KINDS = tuple(MODEL_PATHS)

for _d in (*(OWN_DIR / k for k in KINDS), *(PROCESSED_DIR / k for k in KINDS), MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- vocabulary
# Static (single-frame) ISL signs: numbers 1-9 and letters A-Z.
LETTERS = [str(i) for i in range(1, 10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Dynamic (moving) ISL words. Labels use lowercase + underscores.
# Labels starting with "_" are background classes: they are trained but never typed.
WORDS = [
    # INCLUDE "Greetings" words: recordings of these add to the public dataset
    "hello", "how_are_you", "alright", "good_morning", "good_afternoon",
    "good_evening", "good_night", "thank_you", "pleased",
    # everyday words for schools, hospitals and counters
    "sorry", "please", "yes", "no", "help", "doctor", "water", "food", "name",
    # background class: hands resting / moving without signing
    "_idle",
]

# ---------------------------------------------------------------- sequences
SEQ_LEN = 30            # frames fed to the word model
STORE_LEN = 60          # frames kept per stored clip (lets training time-warp)
WORD_WINDOW_SECONDS = 2.0
# Sentences are longer, so they are stored and fed to the model with more frames.
SENT_SEQ_LEN = 48
SENT_STORE_LEN = 96


def seq_len(kind: str) -> int:
    return SENT_SEQ_LEN if kind == "sentence" else SEQ_LEN


def store_len(kind: str) -> int:
    return SENT_STORE_LEN if kind == "sentence" else STORE_LEN


# ---------------------------------------------------------------- live tuning
LETTER_MIN_CONFIDENCE = 0.70   # below this a letter is shown but not typed
LETTER_HOLD_SECONDS = 0.8      # how long a letter must be held to be typed
LETTER_SMOOTHING = 0.35        # weight of the newest frame when averaging predictions (1 = no smoothing)
SPACE_AFTER_SECONDS = 1.5      # hands away this long -> insert a space
WORD_MIN_CONFIDENCE = 0.70
WORD_PREDICT_EVERY = 0.25      # seconds between word-model predictions
WORD_COOLDOWN_SECONDS = 1.0    # minimum gap between two typed words
WORD_AGREE = 3                 # predictions in a row that must agree before a word is typed
# Sentences: the signer raises the hands, signs the whole sentence, then lowers them.
SENTENCE_END_PAUSE = 0.7       # hands out of view this long ends the sentence
SENTENCE_MIN_SECONDS = 1.0     # shorter movements are ignored
SENTENCE_MAX_SECONDS = 12.0
SENTENCE_MIN_CONFIDENCE = 0.60  # a sentence is only typed when the model is this sure...
SENTENCE_MIN_MARGIN = 0.25      # ...and clearly ahead of the next best sentence
SENTENCE_UNSURE_CONFIDENCE = 0.25  # between this and the minimum: "Did you mean ...?" buttons
SENTENCE_MIN_TRAVEL = 0.12      # wrists must travel at least this far (fraction of frame) to count as signing
SENTENCE_STILL_SECONDS = 1.0   # ...or the hands stay still this long (signer keeps them in view)
SENTENCE_STILL_MOVE = 0.025    # wrist movement below this (fraction of the frame height) counts as still


def display_label(label: str) -> str:
    """'thank_you' -> 'thank you'; letters stay as they are."""
    return label.replace("_", " ")


def is_background(label: str) -> bool:
    return label.startswith("_")
