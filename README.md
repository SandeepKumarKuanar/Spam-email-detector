# Spam Message Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Django](https://img.shields.io/badge/Django-5.2-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![Model](https://img.shields.io/badge/Model-LogisticRegression-green.svg)
![Deployed on Render](https://img.shields.io/badge/Deployed-Render-purple.svg)

An end-to-end spam message classification web application using TF-IDF, Logistic Regression, a Django REST API, and a vanilla JavaScript frontend. The trained model classifies short messages (SMS-style text) as **spam** or **ham**.

> **Live demo:** <a href="https://sandeepkumarkuanar.github.io/Spam-email-detector/" target="_blank">sandeepkumarkuanar.github.io/Spam-email-detector</a>
>
> **API:** <a href="https://spam-email-detector-1-843g.onrender.com/api/predict/" target="_blank">spam-email-detector-1-843g.onrender.com/api/predict/</a>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works: The ML Pipeline](#how-it-works-the-ml-pipeline)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [API Reference](#api-reference)
- [Deployment Architecture](#deployment-architecture)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Documentation Note](#documentation-note)
- [Author](#author)
- [License](#license)

---

## Overview

This project packages a classic text-classification problem — deciding whether a short message is spam or legitimate — as a full-stack application:

- **ML core:** a `scikit-learn` Logistic Regression classifier trained on 5,572 SMS messages from the SMS Spam Collection dataset using TF-IDF vectorization. The trained model and vectorizer are persisted as `.joblib` artifacts.
- **Backend:** a Django REST Framework API that loads both artifacts once at app startup and serves predictions as JSON.
- **Frontend:** a single-page vanilla HTML/CSS/JS UI that posts a message to the API and renders the spam/ham verdict.

The prediction flow does not use the Django database; it runs entirely in memory on the pre-loaded model and vectorizer.

---

## Features

- **Logistic Regression + TF-IDF** — an interpretable pipeline (`sklearn.linear_model.LogisticRegression`, default hyperparameters) that runs without a GPU.
- **Eager model loading** — the model and vectorizer are loaded once when the Django app starts (`api/apps.py`), so every request hits a warm model.
- **CORS-restricted JSON API** — CORS is configured only for the GitHub Pages origin `https://sandeepkumarkuanar.github.io` (see `CORS_ALLOWED_ORIGINS` in `spam_project/settings.py`), so browsers block calls from any other origin.
- **Dependency-free frontend** — a single-page UI in plain HTML, CSS, and JavaScript (no build step, no framework).
- **Persisted model artifacts** — `model/spam_detector_model.joblib` and `model/tfidf_vectorizer.joblib` let the model be redeployed without retraining.

---

## Screenshots

*The live web UI — type a message, get a verdict.*

![Spam message classifier interface](screenshots/landing.png)

---

## Tech Stack

| Layer       | Technology                                                                       |
|-------------|----------------------------------------------------------------------------------|
| Language    | Python 3.12                                                                      |
| ML          | scikit-learn (`LogisticRegression`, `TfidfVectorizer`), `joblib` for persistence  |
| Data        | pandas (`train_model.py` only)                                                   |
| Backend     | Django 5.2, Django REST Framework, django-cors-headers                            |
| Frontend    | Vanilla HTML5, CSS3, JavaScript (no framework)                                   |
| Deployment  | Render Web Service (Gunicorn as the WSGI server) · GitHub Pages (static frontend) |
| Versioning  | Git, with model artifacts tracked as binaries                                    |

Dependencies are declared unpinned in `requirements.txt`.

---

## Dataset

The dataset is the **SMS Spam Collection** (`model/dataset/spam.csv`), a public dataset of SMS messages widely used for spam classification research. It is not an email dataset.

- **5,572** messages × **2** columns (`Category`, `Message`)
- Binary labels: `ham` (legitimate) and `spam`
- Class distribution is **imbalanced**: **4,825** ham (~86.6%) vs. **747** spam (~13.4%), which matters when interpreting the accuracy figure below.

---

## How It Works: The ML Pipeline

Training logic lives in [`train_model.py`](train_model.py):

1. **Load & clean** — `model/dataset/spam.csv` is read; missing values are replaced with empty strings.
2. **Encode labels** — `ham` is mapped to `0` and `spam` to `1`.
3. **Split** — 80/20 train/test split (`test_size=0.2`, `random_state=3`).
4. **Vectorize** — a `TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)` is fit on the training messages.
5. **Train** — a `LogisticRegression` (default hyperparameters) is fit on the TF-IDF features.
6. **Evaluate** — test accuracy is printed to the console.
7. **Persist** — the trained model and vectorizer are saved with `joblib` to `model/`.

The **Django API** mirrors this at serving time. `api/apps.py` loads both `.joblib` files into memory when the app starts, and `api/views.py` transforms each incoming message with the same vectorizer before calling `model.predict()`, guaranteeing train/serve consistency.

**Request/response flow:**

1. A user types a message in `index.html`.
2. `scripts/main.js` POSTs `{"message": "..."}` to `/api/predict/` via `fetch`.
3. `PredictView.post` reads the `message` field; if it is empty or missing, it returns `400 {"error": "Message not provided"}`.
4. Otherwise the message is vectorized and passed to the model, returning `200 {"prediction": "spam"}` or `200 {"prediction": "ham"}`.
5. The frontend renders the verdict in a modal.

---

## Model Performance

| Dataset | Accuracy |
|---------|----------|
| **Test** | ≈ **0.967** |

This is a **single hold-out accuracy** of ≈0.967 (≈96.7%) on the fixed 20% test split (1,115 messages, `random_state=3`). It is the only metric the current implementation computes.

Because the dataset is ~86.6% ham, a "predict everything as ham" baseline already scores ~0.87. This single accuracy figure should not be read as proof of production-quality spam detection; see [Known Limitations](#known-limitations).

---

## Project Structure

```
spam-email-classifier/
├── api/                              # Django app exposing the prediction API
│   ├── apps.py                       # Loads model + vectorizer at app startup
│   ├── urls.py                       # /api/predict/ route
│   ├── views.py                      # PredictView API endpoint
│   ├── tests.py                      # Empty stub (no tests)
│   └── migrations/
├── spam_project/                     # Django project config (settings, urls, wsgi/asgi)
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── model/                            # ML artifacts
│   ├── dataset/
│   │   └── spam.csv                  # SMS Spam Collection training data
│   ├── main.py                       # Standalone experimentation script (not used by the app)
│   ├── spam_detector_model.joblib    # Trained LogisticRegression classifier
│   └── tfidf_vectorizer.joblib       # Trained TF-IDF vectorizer
├── screenshots/
│   └── landing.png                   # Web UI screenshot
├── scripts/
│   └── main.js                       # Frontend JS (API client + UI handling)
├── styles/
│   └── main.css                      # Frontend styles
├── index.html                        # Web frontend (single page)
├── train_model.py                    # ML pipeline: train, evaluate, save
├── manage.py                         # Django management entrypoint
├── requirements.txt                  # Python dependencies (unpinned)
├── db.sqlite3                        # Django SQLite DB (not used by the prediction flow)
├── README.md
├── LICENSE
└── .gitignore
```

---

## Local Setup

### Prerequisites

- Python 3.12 or newer
- `pip`

### Steps

1. **Clone the repository** (from the repo root, not this directory name, after cloning):

   ```bash
   git clone git@github.com:SandeepKumarKuanar/Spam-email-detector.git
   cd Spam-email-detector
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Retrain the model**

   ```bash
   python train_model.py
   ```

   This re-splits the dataset, retrains the classifier, prints the hold-out test accuracy, and overwrites `model/spam_detector_model.joblib` and `model/tfidf_vectorizer.joblib`. Run it from the repository root because it uses relative paths. Note: the script currently fails on newer pandas releases (see [Known Limitations](#known-limitations)).

5. **Run the Django API**

   ```bash
   python manage.py runserver
   ```

   Before this works locally, add `127.0.0.1` and `localhost` to `ALLOWED_HOSTS` in `spam_project/settings.py` (it currently only lists the Render hostname). The API is then available at `http://127.0.0.1:8000/api/predict/`.

6. **Open the frontend**

   Open `index.html` in a browser. It is a static page that calls the API via CORS, but the API only allows the GitHub Pages origin, and the API URL is hardcoded to the production Render endpoint in [`scripts/main.js`](scripts/main.js):45. To run the frontend end-to-end locally you must also add your local origin to `CORS_ALLOWED_ORIGINS` and update the hardcoded URL.

---

## API Reference

Base URL: `https://spam-email-detector-1-843g.onrender.com/api/` (or `http://127.0.0.1:8000/api/` locally)

### `POST /api/predict/`

Classifies a short message as `spam` or `ham`.

**Request:**
```json
{
  "message": "Congratulations! You won a free iPhone. Click here to claim."
}
```

**Example:**
```bash
curl -X POST https://spam-email-detector-1-843g.onrender.com/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Winner! Click here to claim your free prize now."}'
```

**Responses:**

| Code | Meaning |
|------|---------|
| `200` | Prediction successful: `{"prediction": "spam"}` or `{"prediction": "ham"}` |
| `400` | No `message` provided: `{"error": "Message not provided"}` |

This is the only endpoint the API implements.

---

## Deployment Architecture

The application is split into two independent hosts:

- **API (Render Web Service):** the Django project is served by Gunicorn with `gunicorn spam_project.wsgi:application`. The model, vectorizer, and dataset ship inside the API repository; no database or external service is required at runtime.
- **Frontend (GitHub Pages):** the static `index.html`, `scripts/main.js`, and `styles/main.css` are served from GitHub Pages at the live demo URL.

The static frontend calls the API over CORS. `CORS_ALLOWED_ORIGINS` in `spam_project/settings.py` contains only `https://sandeepkumarkuanar.github.io`, so browsers will block API calls from any other origin. `ALLOWED_HOSTS` contains only `spam-email-detector-1-843g.onrender.com`.

From the Render dashboard:

1. **Build command**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start command**
   ```bash
   gunicorn spam_project.wsgi:application
   ```
3. **`ALLOWED_HOSTS`** — add the Render hostname to `settings.py` if the domain changes, and add `127.0.0.1`/`localhost` for local development.
4. **Environment variables** — `SECRET_KEY` and `DEBUG` are currently set directly in `settings.py` (`DEBUG = True`). For a hardened deployment, move both to environment variables (`DEBUG=False`) and set them from the Render dashboard. See [Known Limitations](#known-limitations).
5. **CORS** — if the UI is deployed on another origin, add it to `CORS_ALLOWED_ORIGINS`; otherwise browsers will block the API calls.

---

## Known Limitations

- **Imbalanced dataset, plain accuracy metric.** The dataset is ~86.6% ham, so a "predict everything as ham" baseline already scores ~0.87. The reported ≈0.967 test accuracy is a single hold-out number (`random_state=3`), with no cross-validation and no precision/recall or confusion-matrix breakdown.
- **Retraining depends on an old pandas behavior.** `train_model.py` assigns integer labels (`0`/`1`) directly into the string-dtype `Category` column. This works on the pandas 1.x line but raises a `TypeError` on current pandas releases, so the documented retrain step cannot run on a fresh install today without a compatibility fix.
- **Debug settings are enabled.** `DEBUG = True` and a hardcoded `SECRET_KEY` in `settings.py` are convenient locally but are not production-safe. `requirements.txt` is also unpinned.
- **Restrictive host/origin allowlists.** `ALLOWED_HOSTS` and `CORS_ALLOWED_ORIGINS` are hardcoded to the current Render hostname and GitHub Pages origin; running locally or serving the UI elsewhere requires editing `settings.py`.
- **Manual retrain flow.** Re-running `train_model.py` overwrites the model files, but the deployed service picks them up only after a redeploy — there is no automated retrain/redeploy pipeline.
- **Hardcoded frontend API URL.** [`scripts/main.js`](scripts/main.js):45 points at the production Render URL; local testing requires editing it by hand. The frontend copy also refers to "email", while the trained model operates on SMS-style messages.
- **No automated tests.** `api/tests.py` is an empty stub — the `/api/predict/` endpoint has no automated coverage.
- **Generated artifacts are still tracked.** Despite the entries in `.gitignore`, `db.sqlite3` and both `.joblib` files were committed before the ignore rules were added and remain under version control.

---

## Roadmap

- **Add automated tests + CI** — cover `/api/predict/` (spam/ham predictions, missing message → 400) with a GitHub Actions workflow.
- **Harden configuration** — move `SECRET_KEY` / `DEBUG` / `ALLOWED_HOSTS` / CORS to environment variables and pin dependency versions.
- **Better evaluation** — add precision/recall and a confusion matrix, try class balancing (e.g., `class_weight`), and compare against alternatives (Naive Bayes, SVM) to close the gap on the majority-class baseline.
- **Fix retraining** — make `train_model.py` compatible with current pandas and document exact dependency versions.
- **Configure the frontend without editing source** — make the API base URL configurable, and align the UI copy with the SMS dataset.
- **Git hygiene** — untrack `db.sqlite3` and the `.joblib` artifacts now that `.gitignore` covers them.

---

## Documentation Note

This README was updated with assistance from OpenCode. The application implementation and code were developed by me; OpenCode was used here only to improve and maintain the project documentation.

---

## Author

**Sandeep Kumar Kuanar**

- Blog / Contact: [sandeepkumarkuanar.pythonanywhere.com/contact](https://sandeepkumarkuanar.pythonanywhere.com/contact)
- GitHub: [SandeepKumarKuanar](https://github.com/SandeepKumarKuanar)
- X: [@kuanar_sandeep](https://x.com/kuanar_sandeep)
- Email: kuanarsandeepkumar@gmail.com

---

## License

[MIT](LICENSE) © 2026 Sandeep Kumar Kuanar