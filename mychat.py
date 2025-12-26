import uuid
import json
from pathlib import Path
import re
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from googletrans import Translator
from gtts import gTTS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ================= FLASK SETUP =================
app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

translator = Translator()

# ================= TTS LANGUAGE MAP =================
TTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "mr": "hi"  # Marathi fallback to Hindi voice
}

# ================= EMBEDDING MODEL =================
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)
EMBED_DIM = model.get_sentence_embedding_dimension()

# ================= FAISS =================
FAISS_INDEX_FILE = Path("faiss_index.index")
META_FILE = Path("vector_id_meta.json")

vector_meta = {}
if META_FILE.exists():
    try:
        vector_meta = json.loads(META_FILE.read_text())
    except:
        pass

if FAISS_INDEX_FILE.exists():
    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
else:
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)

# ================= TEXT UTILITIES =================
def clean_sentence(text):
    text = re.sub(r"(figure|fig)\s*\d+", "", text, flags=re.I)
    text = re.sub(r"\b\d+(\.\d+)?\b", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def embed_text(text):
    vec = model.encode(text)
    return (vec / (np.linalg.norm(vec) + 1e-9)).astype("float32")

def strip_html_and_emoji_for_tts(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(
        "[" "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF" "]", "", text
    )
    return re.sub(r"\s{2,}", " ", text).strip()

# ================= PDF PROCESSING =================
def extract_pdf_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

def add_to_faiss(paragraphs):
    vectors = []
    for p in paragraphs:
        vid = str(uuid.uuid4())
        vector_meta[vid] = {"text": p}
        vectors.append(embed_text(p))
    if vectors:
        faiss_index.add(np.vstack(vectors))
        faiss.write_index(faiss_index, str(FAISS_INDEX_FILE))
        META_FILE.write_text(json.dumps(vector_meta, ensure_ascii=False))

# ================= SUMMARIZER =================
def summarize(text):
    lines = re.split(r'(?<=[.!?])\s+', text)
    lines = [clean_sentence(l) for l in lines if len(l.strip()) > 3]
    bullets = lines[:4]
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"

# ================= EXISTING GREETINGS (UNCHANGED) =================
GENERAL_GREETINGS = {
    "hi": "Hello! 😊 How can I help you?",
    "hii": "Hi! 😊 How can I assist you?",
    "hello": "Hello! 👋 How can I help you?",
    "namaste": "🙏 Namaste! Aapki madat kaise kar sakta hoon?",
    "namaskar": "🙏 Namaskar! How can I help you?",
    "suprabhat": "🌅 Suprabhat! Kaise sahayata karu?"
}

TIME_GREETINGS = {
    "good morning": "🌅 Good morning! How can I help you?",
    "good afternoon": "☀️ Good afternoon! How may I assist?",
    "good evening": "🌇 Good evening! What can I do for you?",
    "good night": "🌙 Good night! Feel free to ask anytime."
}

MULTI_LANG_GREETINGS = {
    "नमस्ते": "namaste",
    "नमस्कार": "namaskar",
    "सुप्रभात": "suprabhat",
    "शुभ प्रभात": "good morning",
    "शुभ संध्या": "good evening",
    "शुभ रात्रि": "good night",
    "शुभ रात्री": "good night"
}

# ================= ADDITION: LANGUAGE-SPECIFIC REPLIES =================
MARATHI_GREETINGS = {
    "namaste": "🙏 नमस्कार! मी तुमची कशी मदत करू शकतो?",
    "namaskar": "🙏 नमस्कार! कृपया तुमचा प्रश्न विचारा.",
    "suprabhat": "🌅 सुप्रभात! तुमचा दिवस शुभ जावो."
}

HINDI_GREETINGS = {
    "namaste": "🙏 नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
    "suprabhat": "🌅 सुप्रभात! आपका दिन शुभ हो."
}

THANK_YOU_WORDS = {
    "en": ["thank you", "thanks"],
    "hi": ["धन्यवाद", "आभार"],
    "mr": ["धन्यवाद", "आभार"]
}

THANK_YOU_REPLY = {
    "en": "You're welcome! 😊 Have a great day!",
    "hi": "आपका स्वागत है! 😊 आपका दिन शुभ हो।",
    "mr": "स्वागत आहे! 😊 तुमचा दिवस छान जावो."
}

# ================= ROUTES =================
@app.route("/")
def index_route():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    files = request.files.getlist("pdf")
    for f in files:
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
        f.save(path)
        add_to_faiss(extract_pdf_paragraphs(path))
    return jsonify({"success": True})

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = (request.json.get("message") or "").strip()
    if not user_msg:
        return jsonify({"success": False})

    try:
        detected_lang = translator.detect(user_msg).lang
    except:
        detected_lang = "en"

    # -------- THANK YOU (ADDED) --------
    for word in THANK_YOU_WORDS.get(detected_lang, []):
        if word in user_msg.lower():
            reply_final = THANK_YOU_REPLY.get(detected_lang, THANK_YOU_REPLY["en"])
            break
    else:
        reply_final = None

    # -------- EXISTING GREETING LOGIC (ENHANCED) --------
    if reply_final is None:
        for native, intent in MULTI_LANG_GREETINGS.items():
            if native in user_msg:
                if detected_lang == "mr":
                    reply_final = MARATHI_GREETINGS.get(intent)
                elif detected_lang == "hi":
                    reply_final = HINDI_GREETINGS.get(intent)
                else:
                    reply_final = TIME_GREETINGS.get(intent) or GENERAL_GREETINGS.get(intent)
                break

    if reply_final is None:
        msg_en = translator.translate(user_msg, dest="en").text if detected_lang != "en" else user_msg
        clean = msg_en.lower()

        for k, v in {**TIME_GREETINGS, **GENERAL_GREETINGS}.items():
            if k in clean:
                reply_final = v
                break

    # -------- FAISS SEARCH (UNCHANGED) --------
    if reply_final is None:
        qv = embed_text(msg_en).reshape(1, -1)
        D, I = faiss_index.search(qv, 1)

        if I[0][0] != -1 and D[0][0] > 0.30:
            vid = list(vector_meta.keys())[I[0][0]]
            reply_final = summarize(vector_meta[vid]["text"])
        else:
            reply_final = "Sorry, this question is outside the context of the uploaded PDFs."

    # -------- TRANSLATE BACK (UNCHANGED) --------
    if detected_lang != "en":
        try:
            reply_final = translator.translate(reply_final, dest=detected_lang).text
        except:
            pass

    # -------- TTS (UNCHANGED) --------
    audio = None
    try:
        tts_text = strip_html_and_emoji_for_tts(reply_final)
        tts = gTTS(tts_text, lang=TTS_LANG_MAP.get(detected_lang, "en"), tld="co.in")
        audio_path = f"static/tts_{uuid.uuid4().hex}.mp3"
        tts.save(audio_path)
        audio = "/" + audio_path
    except:
        pass

    return jsonify({"success": True, "text": reply_final, "audio": audio})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
