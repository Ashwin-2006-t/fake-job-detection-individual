# JobGuard AI – Fake Job Posting Detection

A beginner‑friendly project that shows how to build an end‑to‑end Machine Learning + Flask web app to classify job postings as “Fake” or “Real”. It includes data cleaning, model training, evaluation charts, and a modern UI with dark/light theme and an admin panel.

---

## 1. What This Project Does
Many scam job postings share patterns: exaggerated salary, vague duties, payment requests. JobGuard AI:
- Cleans raw job descriptions (HTML removal, lowercasing, stopword removal, lemmatization).
- Converts text to numerical features using TF‑IDF.
- Trains several models (Logistic Regression, Decision Tree, Random Forest).
- Saves the best model + vectorizer for reuse.
- Serves a Flask web interface for interactive predictions.
- Logs each prediction in a SQLite database for history and analytics.
- Provides an admin dashboard with charts (fake vs real distribution + daily volume).

---

## 2. Who Is This For?
Someone new to:
- Python data science (pandas, scikit‑learn, nltk)
- Web apps (Flask basics: routes, templates, sessions)
- Persisting ML results (SQLite)
- Simple UI/UX improvements (HTML/CSS + theme toggle)

---

## 3. Core Files
| File | Purpose |
|------|---------|
| `fake_job_pipeline.py` | Train + evaluate models, produce plots, save artifacts |
| `app.py` | Flask server (login, prediction, history, dashboard) |
| `templates/` | Jinja2 HTML pages (index, result, history, dashboard, login) |
| `job_predictions.db` | SQLite database (auto‑created) |
| `fake_job_model.pkl` / `tfidf_vectorizer.pkl` | Saved model artifacts |
| `requirements.txt` | Dependencies list |

---

## 4. Prerequisites
Install Python 3.11+ (or 3.10). Then:
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate (macOS/Linux)
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
Flask
joblib
scikit-learn
pandas
numpy
nltk
beautifulsoup4
matplotlib
```

---

## 5. Train the Model
Run once to build artifacts:
```bash
python fake_job_pipeline.py
```
Outputs:
- `fake_job_model.pkl`
- `tfidf_vectorizer.pkl`
- Evaluation plots: ROC curves, feature importance, accuracy comparisons.

If you get NLTK download errors, ensure internet access for first run (script calls `nltk.download()`).

---

## 6. Start the Web App
After training:
```bash
python app.py
```
Visit: `http://127.0.0.1:5000/`

Default admin credentials auto‑inserted:
```
username: admin
password: admin123
```
(For real use, change or hash the password.)

---

## 7. Using the Interface
1. Log in (admin).
2. Paste a job description.
3. Submit to get:
   - Label (Fake / Real)
   - Confidence %
4. View Recent Predictions on History page.
5. Open Dashboard for charts and totals.
6. Switch Light/Dark theme using the header toggle.

Input validation:
- Minimum 5 words.
- Rejects mostly non‑alphabetic strings.
- Basic cleanup already applied during training stage.

---

## 8. How Predictions Work
- Text → TF‑IDF vector
- Logistic Regression (or chosen model) → probability
- Probability threshold (0.5) → label
Confidence shown as probability for the predicted class.

---

## 9. Database Schema (SQLite)
`predictions` table:
| Field | Type | Notes |
|-------|------|-------|
| id | INTEGER | Auto increment |
| job_description | TEXT | Raw text submitted |
| prediction | TEXT | "Fake Job" / "Real Job" |
| confidence | REAL | Percent (0–100) |
| timestamp | DATETIME | Auto default |

`admin` table stores plain credentials (demo only).

---

## 10. Common Issues (Quick Fixes)
| Problem | Fix |
|---------|-----|
| Model file missing | Re-run `fake_job_pipeline.py` |
| NLTK errors | Ensure internet; run script again |
| Unicode errors | Save files as UTF‑8 |
| “Can’t pickle lambda” | Replace inline lambdas with named functions (already handled) |
| Long GridSearch time | Reduce hyperparameter grid or skip |

---

## 11. Extending the Project
Ideas:
- Add password hashing (`werkzeug.security`).
- Deploy via Docker/Gunicorn.
- Provide a REST JSON endpoint (`/api/predict`).
- Add user roles.
- Allow batch CSV uploads for scoring.
- Experiment with modern NLP (e.g., DistilBERT) for improved accuracy.

---

## 12. Quick CLI Example
```python
import joblib
model = joblib.load('fake_job_model.pkl')
vec = joblib.load('tfidf_vectorizer.pkl')
text = "We offer high salary. Send payment to proceed..."
X = vec.transform([text])
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]
print(pred, prob)
```

---

## 13. Disclaimer
This is a learning/demo tool. False positives/negatives can occur. Always manually review suspicious postings.

---

## 14. Summary
You learn:
- Text preprocessing pipeline.
- Model training and evaluation.
- Saving/loading ML artifacts.
- Flask session‑based auth.
- SQLite integration.
- Building a clean, theme‑aware UI.

Start small: train → run app → predict → explore dashboard. Then iterate.

Enjoy building with JobGuard AI.
