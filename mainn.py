# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"F:\Ocr\tesseract.exe"
import os   #work with file paths and folders
import io   #handle files in memory (bytes) instead of disk
import re   #Used for text pattern matching
import numpy as np   #Used for arrays and math operations &images are stored as pixel arrays
import webbrowser, threading  #opens browser automatically &  runs something in background
import cv2      #image processing & feature detection
from PIL import Image   #used to open and manipulate images
from skimage.filters import sobel   #used to detect edges
from scipy.signal import convolve2d   #Used to apply convolution (filters) on image
#import pytesseract   #OCR tool
from flask import Flask, request, jsonify, send_from_directory  #web server

# ================= APP INIT =================
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Used to load index.html

# ================= CONFIG =================
CONFIG = {
    "RESIZE": 512,

    "WEIGHTS": {
        "ela": 0.05,
        "noise": 0.05,
        "edge": 0.05,
        "color": 0.10,
        "clone": 0.15,
        "forgery": 0.15,
        "text": 0.10,
        "amount": 0.10,
        "semantic": 0.25,
    },

    "THRESHOLD_REAL": 0.65,
    "THRESHOLD_FAKE": 0.40,
    "SCREENSHOT_BIAS": 0.02,
}

# ================= TEXT CLEANING =================
#Function to clean extracted text
# Convert to lowercase
# Remove special characters
# Remove extra spaces
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    return text.strip()

# ================= FREQUENCY MODEL =================
# Replace hardcoding with probability-based logic
WORD_FREQUENCY = {
    "easypaisa": 0.95,
    "jazzcash": 0.85,
    "transaction successful": 0.90,
    "payment received": 0.88,
    "ubl digital": 0.10,   # rare suspicious automatically
}
#Calculates suspicion based on words
#Rare word then high score then more suspicious
def frequency_score(text):
    score = 0.0
    for word, freq in WORD_FREQUENCY.items():
        if word in text:
            score += (1 - freq)
    return min(score, 1.0)

# ================= PREPROCESS =================
def preprocess(img_np):
    img = cv2.resize(img_np, (CONFIG["RESIZE"], CONFIG["RESIZE"]))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, gray

def is_screenshot(img_np):
    h, w = img_np.shape[:2]
    return (h / w) > 1.4 #Tall image so likely screenshot

# ================= FORENSIC MODULES =================

def ela_score(img_pil): #Compression check
    try:
        img_rgb = img_pil.convert("RGB")   # ensure no alpha channel
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        rec = Image.open(buf).convert("RGB")
        diff = np.abs(np.array(img_rgb).astype(np.float32) - np.array(rec).astype(np.float32))
        return float(np.clip(diff.mean() / 40, 0, 1))
    except Exception:
        return 0.1

def noise_score(gray):
    try:
        k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        noise = np.abs(convolve2d(gray.astype(np.float32), k, mode="same")) #Apply Laplacian filter
        return float(np.clip(np.std(noise) / 40, 0, 1)) #Measure variation
    except Exception:
        return 0.1

def edge_score(gray):
    try:
        edges = sobel(gray / 255.0) #Detect edges
        return float(np.clip(np.std(edges) * 2.5, 0, 1))
    except Exception:
        return 0.1

def color_score(rgb):
    try:
        h, w = rgb.shape[:2]
        q1 = rgb[:h//2, :w//2]
        q2 = rgb[h//2:, w//2:]
        return float(np.clip(abs(q1.mean() - q2.mean()) / 40, 0, 1)) #Compare two parts of image
    except Exception:
        return 0.1

def clone_score(gray):
    orb = cv2.ORB_create() #Find keypoints
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 15:
        return 0.2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Match features inside same image
    matches = bf.match(des, des)

    suspicious = 0
    for m in matches:
        if m.queryIdx == m.trainIdx:
            continue
        p1 = np.array(kp[m.queryIdx].pt)
        p2 = np.array(kp[m.trainIdx].pt)
        if np.linalg.norm(p1 - p2) > 50 and m.distance < 30:
            suspicious += 1

    return float(np.clip(suspicious / len(kp) * 2.5, 0, 1))

def forgery_score(gray):
    try:
        dct = cv2.dct(gray.astype(np.float32)/255.0)  #Convert to frequency
        hi = np.mean(np.abs(dct[128:,128:]))
        lo = np.mean(np.abs(dct[:32,:32])) + 1e-6
        return float(np.clip((hi/lo) * 4, 0, 1)) #Compare high vs low frequency
    except Exception:
        return 0.1

# ================= TEXT ANALYSIS =================

def extract_text(gray):
    try: 
        text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
        return clean_text(text)
    except:
        return ""

def extract_raw_text(gray):
    """Better OCR for timestamp detection"""
    try:
        return pytesseract.image_to_string(
            gray,
            config="--oem 3 --psm 6"
        )
    except:
        return ""

def text_uniformity_score(gray):
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    heights = []
    for i in range(len(data["text"])):
        if data["text"][i].strip(): #Checks if font size is consistent
            heights.append(data["height"][i])

    if len(heights) < 5:
        return 0.2

    return float(np.clip(np.std(heights) / (np.mean(heights)+1e-6), 0, 1))

def amount_region_score(gray):
    try:
        h, w = gray.shape
        region = gray[int(h*0.6):int(h*0.8), int(w*0.2):int(w*0.9)] #Crop area where amount is expected
        sharpness = cv2.Laplacian(region, cv2.CV_64F).var() # Blur = suspicious
        return float(np.clip(1.0 - sharpness/800, 0, 1))
    except Exception:
        return 0.1

# ================= RECIPIENT CHECK =================
# Returns True if "Abdullah Kamran" is found in OCR text, False otherwise
def check_recipient(text):
    normalized = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    # Match "abdullah" and "kamran" appearing within 40 chars of each other
    match = re.search(r'abdullah.{0,40}kamran|kamran.{0,40}abdullah', normalized)
    return match is not None

# ================= TRANSACTION TIME CHECK =================
# Extracts the transaction TIMESTAMP from OCR text and compares it to NOW.
# Returns (diff_minutes or None, warning_string or None, tx_time_str or None)
#
# Uses only Python built-in datetime — no extra installs needed.
# PKT = UTC+5 (Pakistan Standard Time, fixed offset, no pytz required)
#
# Supported OCR timestamp formats (Pakistani payment apps):
#   "10:53 PM"  /  "10:53:22 PM"  (time only — uses today's date)
#   "26 Apr 2026, 10:53 PM"
#   "Apr 26, 2026 10:53 PM"
#   "2026-04-26 10:53"
#   "26/04/2026 10:53"
from datetime import datetime, timezone, timedelta

# PKT = UTC+5, no pytz needed
PKT = timezone(timedelta(hours=5))

def check_transaction_time(text):
    now_pkt = datetime.now(PKT)
    tx_dt = None

    MONTHS = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    }

    def parse_12h_time(hh, mm, ss, ampm):
        h, m, s = int(hh), int(mm), int(ss) if ss else 0
        if ampm and ampm.lower() == "pm" and h != 12:
            h += 12
        elif ampm and ampm.lower() == "am" and h == 12:
            h = 0
        return h, m, s

    def make_dt(year, month, day, h, m, s):
        return datetime(int(year), int(month), int(day), h, m, s, tzinfo=PKT)

    # Time separator: colon OR dot (OCR often reads "10.53" instead of "10:53")
    T = r'[:\.]'

    # ── Pattern 1a: "26 Apr 2026, 10:53 PM"  /  "26 Apr 2026, 10.53.48 PM" ────
    p1 = re.search(
        r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})'
        r'[,\s]+(\d{1,2})' + T + r'(\d{2})(?:' + T + r'(\d{2}))?\s*(am|pm)',
        text, re.IGNORECASE)
    if p1:
        day, mon_str, year, hh, mm, ss, ampm = p1.groups()
        h, m, s = parse_12h_time(hh, mm, ss, ampm)
        try: tx_dt = make_dt(year, MONTHS[mon_str[:3].lower()], day, h, m, s)
        except Exception: pass

    # ── Pattern 1b: "Apr 26, 2026 10:53 PM"  /  "Apr 26, 2026 10.53.48 PM" ────
    if not tx_dt:
        p1b = re.search(
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})'
            r'[,\s]+(\d{1,2})' + T + r'(\d{2})(?:' + T + r'(\d{2}))?\s*(am|pm)',
            text, re.IGNORECASE)
        if p1b:
            mon_str, day, year, hh, mm, ss, ampm = p1b.groups()
            h, m, s = parse_12h_time(hh, mm, ss, ampm)
            try: tx_dt = make_dt(year, MONTHS[mon_str[:3].lower()], day, h, m, s)
            except Exception: pass

    # ── Pattern 2a: ISO  "2026-04-26 10:53"  /  "2026-04-26T10.53" ─────────────
    if not tx_dt:
        p2 = re.search(
            r'(\d{4})-(\d{2})-(\d{2})[T\s]+(\d{1,2})' + T + r'(\d{2})(?:' + T + r'(\d{2}))?\s*(am|pm)?',
            text, re.IGNORECASE)
        if p2:
            year, mo, day, hh, mm, ss, ampm = p2.groups()
            h, m, s = parse_12h_time(hh, mm, ss, ampm)
            try: tx_dt = make_dt(year, mo, day, h, m, s)
            except Exception: pass

    # ── Pattern 2b: slash date  "26/04/2026 10:53 PM"  /  "26/04/2026 10.53 PM" ─
    if not tx_dt:
        p2b = re.search(
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})[,\s]+'
            r'(\d{1,2})' + T + r'(\d{2})(?:' + T + r'(\d{2}))?\s*(am|pm)?',
            text, re.IGNORECASE)
        if p2b:
            a, b, year, hh, mm, ss, ampm = p2b.groups()
            h, m, s = parse_12h_time(hh, mm, ss, ampm)
            try: tx_dt = make_dt(year, b, a, h, m, s)   # DD/MM/YYYY
            except Exception: pass

    # ── Pattern 3: time-only with AM/PM  "10:53 PM"  /  "10.53.48 PM" ──────────
    if not tx_dt:
        p3 = re.search(
            r'\b(\d{1,2})' + T + r'(\d{2})(?:' + T + r'(\d{2}))?\s*(am|pm)\b',
            text, re.IGNORECASE)
        if p3:
            hh, mm, ss, ampm = p3.groups()
            h, m, s = parse_12h_time(hh, mm, ss, ampm)
            try: tx_dt = make_dt(now_pkt.year, now_pkt.month, now_pkt.day, h, m, s)
            except Exception: pass

    # ── Pattern 4: 24-h time only  "22:53" / "22.53"  (fallback, assume today) ─
    if not tx_dt:
        p4 = re.search(r'\b([01]\d|2[0-3])' + T + r'([0-5]\d)(?:' + T + r'([0-5]\d))?\b', text)
        if p4:
            hh, mm, ss = p4.groups()
            try: tx_dt = make_dt(now_pkt.year, now_pkt.month, now_pkt.day,
                                 int(hh), int(mm), int(ss) if ss else 0)
            except Exception: pass

    # ── Compute difference vs current PKT time ────────────────────────────────
# EXTRA fallback: detect 4-digit time like 1053 → 10:53
    if tx_dt is None:
        p_extra = re.search(r'\b(\d{2})(\d{2})\b', text)
        if p_extra:
            hh, mm = p_extra.groups()
            try:
                tx_dt = make_dt(now_pkt.year, now_pkt.month, now_pkt.day, int(hh), int(mm), 0)
            except:
                pass

    if tx_dt is None:
        return None, None, None

    tx_time_str = tx_dt.strftime("%d %b %Y %I:%M %p")
    diff_seconds = abs((now_pkt - tx_dt).total_seconds())
    diff_minutes = round(diff_seconds / 60, 1)

    if diff_minutes > 3:
        warning = (
            f"Transaction was made at {tx_time_str} PKT — "
            f"{diff_minutes} min(s) ago (greater than 3 minutes)"
        )
        return diff_minutes, warning, tx_time_str
    else:
        return diff_minutes, None, tx_time_str

# ================= SEMANTIC =================
# Checks:
# Word frequency
# Large amounts
# repeated characters
# missing "transaction successful"
# Recipient must be Abdullah Kamran
def semantic_fraud_score(gray):
    text = extract_text(gray)

    score = 0.0

    # Frequency-based detection
    score += frequency_score(text)

    # Large amount detection
    amounts = re.findall(r'\d{1,3}(?:,\d{3})+', text)
    for amt in amounts:
        val = int(amt.replace(",", ""))
        if val > 300000:
            score += 0.3

    # Suspicious patterns
    if re.search(r'(.)\1{3,}', text):
        score += 0.2

    if "transaction successful" not in text:
        score += 0.2

    # Recipient check — heavily penalise if not sent to Abdullah Kamran
    if not check_recipient(text):
        score += 0.4

    return min(score, 1.0)

# ================= DECISION =================

def decision(scores, screenshot):
    W = CONFIG["WEIGHTS"]

    # Convert to "realness"
    inv = {k: 1 - v for k, v in scores.items()} #Convert fake into real

    total_weight = sum(W.values())
    final = sum(inv[k] * W[k] for k in W) / total_weight

    penalty = 0
#Reduce score if suspicious
    if scores["clone"] > 0.5: penalty += 0.15
    if scores["forgery"] > 0.5: penalty += 0.15
    if scores["semantic"] > 0.5: penalty += 0.30
    if scores["text"] > 0.5: penalty += 0.10

    final = np.clip(final - penalty, 0, 1) #Keep between 0–1

    if screenshot:
        final += CONFIG["SCREENSHOT_BIAS"]

    final = np.clip(final, 0, 1)

    if scores["semantic"] > 0.7:
        return "FAKE", final

    if final >= CONFIG["THRESHOLD_REAL"]:
        return "REAL", final
    elif final <= CONFIG["THRESHOLD_FAKE"]:
        return "FAKE", final
    else:
        return ("REAL" if final > 0.5 else "FAKE"), final

# ================= PIPELINE =================
# Full process:
# Load image
# Convert to array
# Preprocess
# Run all modules
# Get scores
# Decision
def analyze_bytes(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except:
        raise ValueError("Invalid image file")

    np_img = np.array(img)
    rgb, gray = preprocess(np_img)

    scores = {
        "ela": ela_score(img),
        "noise": noise_score(gray),
        "edge": edge_score(gray),
        "color": color_score(rgb),
        "clone": clone_score(gray),
        "forgery": forgery_score(gray),
        "text": text_uniformity_score(gray),
        "amount": amount_region_score(gray),
        "semantic": semantic_fraud_score(gray),
    }

    screenshot = is_screenshot(np_img)
    verdict, final = decision(scores, screenshot)

    # ── Recipient check (uses cleaned text — colons not needed)
    ocr_text = extract_text(gray)
    recipient_valid = check_recipient(ocr_text)
    if not recipient_valid:
        verdict = "FAKE"  # Force FAKE if money not sent to Abdullah Kamran

    # ── Transaction time check (uses RAW text — colons and dots must be preserved)
    raw_ocr_text = extract_raw_text(gray)
    print("OCR DEBUG TEXT:\n", raw_ocr_text)
    tx_duration, tx_warning, tx_time_str = check_transaction_time(raw_ocr_text)

    # No timestamp in receipt = suspicious (real receipts always have date/time)
    if tx_time_str is None:
        verdict = "FAKE"
        tx_warning = "No timestamp found — real transaction receipts always contain a date and time"

    return {
        "verdict": verdict, #Output result
        "confidence": round(final, 4),
        "screenshot": screenshot,
        "details": {k: round(v, 4) for k, v in scores.items()},
        "recipient_valid": recipient_valid,
        "recipient_check": "Sent to Abdullah Kamran \u2713" if recipient_valid else "FAKE \u2014 Money not sent to Abdullah Kamran",
        "transaction_time_minutes": tx_duration,
        "transaction_time_warning": tx_warning,
        "transaction_time_str": tx_time_str,
        "ocr_debug": raw_ocr_text[:300] if raw_ocr_text else "",  # first 300 chars for debug
    }

# ================= ACCURACY TESTING =================
def evaluate_accuracy(dataset_path="dataset"):
    total = 0
    correct = 0

    for label in ["real", "fake"]:
        folder = os.path.join(dataset_path, label)

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(folder, file)

            try:
                with open(path, "rb") as f:
                    result = analyze_bytes(f.read())

                predicted = result["verdict"]
                actual = label.upper()

                total += 1

                if predicted == actual:
                    correct += 1
                    print(f"✅ {file} → CORRECT ({predicted})")
                else:
                    print(f"❌ {file} → WRONG (Predicted: {predicted}, Actual: {actual})")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    if total == 0:
        print("No images found!")
        return

    accuracy = (correct / total) * 100

    print("\n==============================")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("==============================\n")

    return accuracy

# ================= ROUTES =================

@app.route("/") #Opens HTML page
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/analyze", methods=["POST"]) #Accepts uploaded image
def analyze_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = f.read()
        result = analyze_bytes(file_bytes)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Analysis failed",
            "detail": traceback.format_exc()
        }), 500

# ================= RUN =================

if __name__ == "__main__": #Runs only when file executed
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    evaluate_accuracy("dataset")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start() #Opens browser automatically
    
    app.run(debug=False) #Start server
