# Fake Financial Screenshot Detection System

**AI Forensics Engine · v2.1** — Multi-layer forensic analysis for detecting tampered or fabricated Pakistani mobile payment screenshots (Easypaisa, JazzCash, UBL Digital).

---

## Overview

This project is a Flask-based web application that uses computer vision, image forensics, and OCR-based semantic analysis to determine whether a transaction screenshot is **real or fake**. It was built as a final-year AI project targeting a specific fraud scenario — verifying that a payment was genuinely sent, to the correct recipient, within an acceptable time window.

The system combines nine independent forensic signals and fuses them into a single weighted confidence score, producing a **REAL** or **FAKE** verdict along with a breakdown of each analysis layer.

---

## Features

- **ELA (Error Level Analysis)** — detects re-compression artifacts that indicate editing
- **Noise Analysis** — measures pixel-level inconsistencies using a Laplacian filter
- **Edge Detection** — uses Sobel filtering to detect unnatural edge patterns
- **Color Consistency** — compares quadrant-level color distribution for signs of splicing
- **Clone Detection** — uses ORB keypoint matching to find copy-pasted regions
- **DCT Forgery Analysis** — compares high vs low frequency DCT coefficients for manipulation
- **Text Uniformity (OCR)** — checks font-size consistency across detected text regions
- **Amount Region Sharpness** — measures blur in the transaction amount area (a common edit target)
- **Semantic NLP Analysis** — checks for suspicious keywords, missing required phrases, and validates the recipient name
- **Recipient Verification** — flags the screenshot as FAKE if the recipient is not "Abdullah Kamran"
- **Transaction Timestamp Check** — extracts the transaction time via OCR and flags if it is more than 3 minutes old
- **Accuracy Evaluation Mode** — runs against a labeled dataset folder to measure model accuracy

---

## Project Structure

```
Fake Financial Screenshot Detection System_AI_Project/
│
├── mainn.py          # Main application — forensic engine + Flask server
├── index.html        # Frontend UI (drag-and-drop screenshot uploader)
└── dataset/          # Accuracy testing dataset
    ├── real/         # Folder of genuine transaction screenshots
    └── fake/         # Folder of fabricated/edited screenshots
```

---

## How It Works

### 1. Image Ingestion
The user uploads a screenshot through the web UI. The backend receives it as raw bytes, converts it to a NumPy array, and resizes it to 512×512 for uniform processing.

### 2. Forensic Module Pipeline
Nine forensic modules each return a suspicion score between 0.0 (genuine) and 1.0 (suspicious):

| Module | What It Checks |
|---|---|
| `ela_score` | Re-compression artifacts (JPEG quality 90 comparison) |
| `noise_score` | Laplacian convolution noise standard deviation |
| `edge_score` | Sobel edge distribution irregularity |
| `color_score` | Quadrant mean color divergence |
| `clone_score` | ORB feature self-matching (copy-paste detection) |
| `forgery_score` | DCT high/low frequency ratio |
| `text_uniformity_score` | OCR bounding-box height variance |
| `amount_region_score` | Laplacian blur variance in amount region |
| `semantic_fraud_score` | Keyword frequency, recipient name, required phrases |

### 3. Weighted Decision
Scores are inverted (high score = more suspicious = lower "realness") and combined using the weights below:

```
Semantic:   25%    Clone:    15%    Forgery:  15%
Color:      10%    Text:     10%    Amount:   10%
ELA:         5%    Noise:     5%    Edge:      5%
```

Additional penalties are applied if clone, forgery, semantic, or text scores exceed 0.5. A small bias (+0.02) is added for tall images identified as mobile screenshots.

### 4. Hard Rules (Override Scoring)
Regardless of the weighted score, the verdict is forced to **FAKE** if:
- The recipient name "Abdullah Kamran" is not found in the OCR text
- No timestamp is detected in the image
- The transaction timestamp is more than **3 minutes** in the past

### 5. Verdict
- `REAL` — confidence ≥ 0.65
- `FAKE` — confidence ≤ 0.40 (or triggered by a hard rule)
- Between 0.40–0.65 — classified by majority side (>0.5 = REAL, else FAKE)

---

## OCR Timestamp Formats Supported

The system handles all common Pakistani payment app timestamp formats:

| Format | Example |
|---|---|
| Day Month Year + 12h time | `26 Apr 2026, 10:53 PM` |
| Month Day Year + 12h time | `Apr 26, 2026 10:53 PM` |
| ISO datetime | `2026-04-26 10:53` |
| Slash date + time | `26/04/2026 10:53 PM` |
| Time only (assumes today) | `10:53 PM` |
| 24-hour time (fallback) | `22:53` |

All timestamps are compared against **Pakistan Standard Time (UTC+5)** using Python's built-in `datetime` module — no `pytz` dependency required.

---

## Installation

### Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system

### Step 1 — Install Python dependencies

```bash
pip install flask pillow opencv-python numpy scikit-image scipy pytesseract
```

### Step 2 — Configure Tesseract path

Open `mainn.py` and update line 3 to point to your Tesseract installation:

```python
# Windows example:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Linux / macOS (usually already on PATH — this line can be removed):
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
```

### Step 3 — Run the app

```bash
python mainn.py
```

The server starts at `http://127.0.0.1:5000` and the browser opens automatically.

---

## Dataset & Accuracy Evaluation

At startup, the system automatically runs an accuracy test against the `dataset/` folder before launching the web server.

**Expected folder structure:**

```
dataset/
├── real/
│   ├── receipt_001.png
│   └── receipt_002.jpg
└── fake/
    ├── edited_001.png
    └── edited_002.jpg
```

Results are printed to the terminal:

```
✅ receipt_001.png → CORRECT (REAL)
❌ edited_002.png  → WRONG (Predicted: REAL, Actual: FAKE)

==============================
Total Images:        20
Correct Predictions: 17
Accuracy:            85.00%
==============================
```

To skip evaluation and go straight to the web server, comment out the `evaluate_accuracy()` call at the bottom of `mainn.py`.

---

## API Reference

### `POST /analyze`

Accepts a multipart form upload and returns a JSON analysis result.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image file>
```

**Response:**
```json
{
  "verdict": "FAKE",
  "confidence": 0.3821,
  "screenshot": true,
  "details": {
    "ela": 0.12,
    "noise": 0.08,
    "edge": 0.21,
    "color": 0.09,
    "clone": 0.55,
    "forgery": 0.61,
    "text": 0.14,
    "amount": 0.33,
    "semantic": 0.75
  },
  "recipient_valid": false,
  "recipient_check": "FAKE — Money not sent to Abdullah Kamran",
  "transaction_time_minutes": 47.3,
  "transaction_time_warning": "Transaction was made at 26 Apr 2026 10:53 PM PKT — 47.3 min(s) ago (greater than 3 minutes)",
  "transaction_time_str": "26 Apr 2026 10:53 PM",
  "ocr_debug": "..."
}
```

---

## Configuration

All tunable parameters live in the `CONFIG` dictionary at the top of `mainn.py`:

| Parameter | Default | Description |
|---|---|---|
| `RESIZE` | `512` | Image resize dimension for processing |
| `THRESHOLD_REAL` | `0.65` | Minimum confidence to classify as REAL |
| `THRESHOLD_FAKE` | `0.40` | Maximum confidence to classify as FAKE |
| `SCREENSHOT_BIAS` | `0.02` | Slight boost for tall mobile screenshots |
| `WEIGHTS` | (see above) | Per-module contribution to final score |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Image Processing | OpenCV, Pillow, scikit-image, SciPy |
| OCR | Tesseract (via pytesseract) |
| Numerical Computing | NumPy |
| Frontend | HTML, CSS, JavaScript (index.html) |

---

## Limitations

- Recipient name is hardcoded to **"Abdullah Kamran"** — this is intentional for the specific use case this system was built for. To adapt it for general use, the recipient check logic in `check_recipient()` should be made configurable.
- OCR accuracy depends on image resolution and Tesseract version. Low-resolution screenshots may cause timestamp or recipient detection to fail.
- The 3-minute transaction window is strict by design. Screenshots shared even a few minutes after a transaction will be flagged with a warning.
- The system is optimized for **Pakistani mobile payment apps** (Easypaisa, JazzCash, UBL Digital). Performance on other receipt types is untested.

---

## Author

**Mustafa Naeem**

