# 📧 Spam Email Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Django](https://img.shields.io/badge/Django-5.2-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![Model](https://img.shields.io/badge/Model-LogisticRegression-green.svg)
![Deployed on Render](https://img.shields.io/badge/Deployed-Render-purple.svg)

An end-to-end machine learning web app that classifies SMS messages as **spam** or **ham** from a trained **Logistic Regression** model, served through a **Django REST API** with a clean, dependency-free web frontend.

> 🔗 **Live demo:** <a href="https://sandeepkumarkuanar.github.io/Spam-email-detector/" target="_blank">sandeepkumarkuanar.github.io/Spam-email-detector</a>
>
> ⚙️ **API:** <a href="https://spam-email-detector-1-843g.onrender.com/api/predict/" target="_blank">spam-email-detector-1-843g.onrender.com/api/predict</a>

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
- [Deployment on Render](#deployment-on-render)
- [Known Issues & Limitations](#known-issues--limitations)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

This project takes a classic text-classification problem — deciding whether a short message is spam or legitimate — and packages it as a complete, production-shaped full-stack application:

- **ML core:** a `scikit-learn` Logistic Regression classifier trained on ~5,500 SMS messages using TF-IDF vectorization, with the trained model and vectorizer persisted as `.joblib` artifacts.
- **Backend:** a Django REST Framework API that loads both artifacts once at app startup and serves predictions as JSON.
- **Frontend:** a single-page vanilla HTML/CSS/JS UI that posts a message to the API and renders the spam/ham verdict.

Everything is wired together so a user can hit the live URL, paste a message, and get a classification back in seconds.

---

## Features

- **Logistic Regression + TF-IDF** — a fast, interpretable pipeline that needs no GPU and classifies in milliseconds.
- **Eager model loading** — the model and vectorizer are loaded once when the Django app starts (`AppConfig`), so every request hits a warm model.
- **CORS-enabled JSON API** — the API is callable from any origin, so the static frontend can live anywhere.
- **Dependency-free frontend** — a small single-page UI built with plain HTML, CSS, and JavaScript (no build step, no framework).
- **Persisted model artifacts** — `spam_detector_model.joblib` and `tfidf_vectorizer.joblib` make the model reproducible and redeployable without retraining.

---

## Screenshots

*The live web UI — drop a message, get a verdict.*

![Spam Email Classifier interface](screenshots/landing.png)

---

## Tech Stack

| Layer       | Technology                                                              |
|-------------|-------------------------------------------------------------------------|
| Language    | Python 3.12                                                             |
| ML          | scikit-learn (LogisticRegression, TfidfVectorizer), joblib for persistence |
| Data        | pandas                                                                  |
| Backend     | Django 5.2, Django REST Framework, django-cors-headers                   |
| Frontend    | Vanilla HTML5, CSS3, JavaScript (no framework)                          |
| Deployment  | Render (Gunicorn as the WSGI server) · GitHub Pages (frontend)          |
| Versioning  | Git, with model artifacts tracked as binaries                           |

---

## Dataset

SMS messages from the **SMS Spam Collection** dataset (`model/dataset/spam.csv`), a public dataset widely used for spam classification.

- **5,572** messages × **2** columns (`Category`, `Message`)
- Binary labels: `ham` (legitimate) and `spam`
- The class distribution is **imbalanced** — roughly **13%** spam vs. **87%** ham, which matters when interpreting the accuracy figure below.

---

## How It Works: The ML Pipeline

The training logic lives entirely in [`train_model.py`](train_model.py):

1. **Load & clean** — `model/dataset/spam.csv` is read; missing values are replaced with empty strings.
2. **Encode labels** — `ham` is mapped to `0` and `spam` to `1`.
3. **Split** — 80/20 train/test split (`random_state=3`).
4. **Vectorize** — a `TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)` is fit on the training messages.
5. **Train** — a `LogisticRegression` classifier is fit on the TF-IDF features.
6. **Evaluate** — test accuracy is printed to the console.
7. **Persist** — the trained model and vectorizer are saved with `joblib` to `model/`.

The **Django API** mirrors this at serving time: `api/apps.py` loads both `.joblib` files into memory when the app boots, and `api/views.py` transforms each incoming message with the same vectorizer before calling `model.predict()` — guaranteeing train/serve consistency.

---

## Model Performance

| Dataset | Accuracy |
|---------|----------|
| **Test** | ≈ **0.96** |

A single hold-out accuracy of **≈0.96** on the 20% test split. Note this is a plain accuracy score over an imbalanced dataset (see [Known Issues & Limitations](#known-issues--limitations)) — a quick sanity check on the live API: a typical "You won a free iPhone" message classifies as `spam`.

---

## Project Structure

```
spam-email-classifier/
├── api/                              # Django app exposing the prediction API
│   ├── apps.py                       # Loads model + vectorizer at app startup
│   ├── urls.py                       # /api/predict/ route
│   ├── views.py                      # PredictView API endpoint
│   └── migrations/
├── spam_project/                     # Django project config (settings, urls, wsgi/asgi)
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── model/                            # ML artifacts
│   ├── dataset/
│   │   └── spam.csv                  # SMS Spam Collection training data
│   ├── spam_detector_model.joblib    # Trained classifier
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
├── requirements.txt                  # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

---

## Local Setup

### Prerequisites

- **Python 3.8+**

### Steps

1. **Clone the repository**

   ```bash
   git clone git@github.com:SandeepKumarKuanar/Spam-email-detector.git
   cd spam-email-classifier
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

   This re-splits the dataset, retrains the classifier, prints test accuracy, and overwrites `model/spam_detector_model.joblib` and `model/tfidf_vectorizer.joblib`.

5. **Run the Django API**

   ```bash
   python manage.py runserver
   ```

   The API is now available at `http://127.0.0.1:8000/api/predict/`.

6. **Open the frontend**

   Open `index.html` in a browser — it's a static page that talks to the API over CORS. For local testing, update the hardcoded `API_BASE_URL` in [`scripts/main.js`](scripts/main.js) to `http://127.0.0.1:8000/api/predict/` before opening the file.

---

## API Reference

Base URL: `https://spam-email-detector-1-843g.onrender.com/api/` (or `http://127.0.0.1:8000/api/` locally)

### `POST /api/predict/`

Classifies a message as `spam` or `ham`.

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

---

## Deployment on Render

The API is deployed as a **Render Web Service**; the frontend is a static page served from GitHub Pages (via CORS).

From the Render dashboard:

1. **Build command**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start command**
   ```bash
   gunicorn spam_project.wsgi:application
   ```
3. **`ALLOWED_HOSTS`** — the Render hostname is listed in [`settings.py`](spam_project/settings.py); add your domain if it changes. For local development, `ALLOWED_HOSTS` should also include `127.0.0.1` and `localhost`.
4. **Environment variables** — `SECRET_KEY` and `DEBUG` are currently set directly in `settings.py` for convenience (with debug left on). For a hardened production setup, move them to environment variables (`DEBUG=False`) and set values from the Render dashboard. See [Known Issues & Limitations](#known-issues--limitations).
5. **CORS** — the frontend origin `https://sandeepkumarkuanar.github.io` is allowlisted in `CORS_ALLOWED_ORIGINS`. Add your own origin if you deploy the UI elsewhere.

---

## Known Issues & Limitations

- **Imbalanced dataset, plain accuracy metric.** The dataset is ~87% ham, so a "predict everything as ham" baseline already scores ~0.87. The reported ≈0.96 test accuracy is a single hold-out number with no cross-validation and no precision/recall breakdown. Precision and recall on the `spam` class would tell a fuller story.
- **Debug settings on for convenience.** `DEBUG = True` and a hardcoded `SECRET_KEY` in `settings.py` make local runs effortless but aren't production-safe. Moving both to environment variables is on the [roadmap](#roadmap).
- **Manual retrain flow.** Re-running `python train_model.py` overwrites the model files, but the deployed service picks them up only after a redeploy — there's no pipeline that retrains and redeploys automatically.
- **Hardcoded frontend URL.** [`scripts/main.js`](scripts/main.js) points at the production Render URL; testing locally requires editing it by hand.
- **Empty test suite.** `api/tests.py` is a stub — the endpoint has no automated tests guarding it.

---

## Roadmap

- **Add automated tests + CI** — cover the `/api/predict/` endpoint (valid spam/ham, missing message → 400) with a GitHub Actions workflow.
- **Harden configuration** — move `SECRET_KEY` / `DEBUG` / `ALLOWED_HOSTS` / CORS to environment variables and pin dependency versions in `requirements.txt`.
- **Better evaluation** — add precision/recall and a confusion matrix, try class balancing (e.g., `class_weight`), and evaluate alternatives (Naive Bayes, SVM) to close the gap on the majority-class baseline.
- **Git hygiene** — stop tracking generated artifacts (`db.sqlite3`, `.joblib` files) now that `.gitignore` is in place.
- **Polish the frontend** — make the success/error states and the local API URL configurable without editing source.
- **Add request logging and rate limiting** — a small but useful step toward production readiness.

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