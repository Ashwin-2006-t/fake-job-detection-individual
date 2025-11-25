import re
import string
import random
import joblib
import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')


from bs4 import BeautifulSoup        # pip install beautifulsoup4
import nltk                          # pip install nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt


# Ensure NLTK resources

nltk_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Utility: text cleaning

def clean_text(text):
    """Return cleaned, tokenized, lemmatized text string."""
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # 4. Remove punctuation and digits
    text = re.sub(r'[\d]', ' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # 5. Tokenize and remove short tokens / stopwords, lemmatize
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = []
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok in STOPWORDS or len(tok) <= 2:
            continue
        tok = lemmatizer.lemmatize(tok)
        cleaned_tokens.append(tok)
    return " ".join(cleaned_tokens)


# Part 1 — Data Understanding

print("=== Part 1: Data Understanding ===")
df = pd.read_csv('fake_job_postings.csv')

# Basic info
print("\nDataset shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# Missing values per column
print("\nMissing values per column:")
print(df.isna().sum())

# Distribution of fraudulent labels
if 'fraudulent' in df.columns:
    print("\nDistribution of 'fraudulent':")
    print(df['fraudulent'].value_counts(normalize=False).to_frame(name='count'))
    print("\nDistribution (proportion):")
    print(df['fraudulent'].value_counts(normalize=True).to_frame(name='proportion'))
else:
    raise KeyError("Column 'fraudulent' not found in dataset.")

# Three short insights (generic — adapt if needed)
print("\n3 Quick Insights (adapt if numbers differ on your data):")
insights = [
    "1) Class imbalance: real jobs typically outnumber fake jobs (check proportions above).",
    "2) Several columns contain missing values (e.g., company profile, salary, or employment_type) — fake posts often have more missing or placeholder info.",
    "3) Some text fields contain HTML/URLs/irrelevant tokens that must be cleaned before modeling."
]
for i in insights:
    print(i)


# Part 2 — Text Cleaning & Preprocessing

print("\n=== Part 2: Text Cleaning & Preprocessing ===")
# Use 'description' column (common in Kaggle dataset)
if 'description' not in df.columns:
    raise KeyError("Column 'description' not found in dataset. Make sure you have the correct CSV.")

# Example raw description for later comparison
example_raw = df['description'].fillna("").iloc[0]

# Create clean_description
print("\nCleaning text (this may take a moment)...")
df['description'] = df['description'].fillna("")
df['clean_description'] = df['description'].apply(clean_text)

# Compare average word count before and after cleaning
def avg_word_count(series):
    return series.apply(lambda t: len(str(t).split())).mean()

avg_before = avg_word_count(df['description'])
avg_after = avg_word_count(df['clean_description'])

print(f"\nAverage word count BEFORE cleaning: {avg_before:.2f}")
print(f"Average word count AFTER cleaning: {avg_after:.2f}")

# Show one raw vs cleaned example
print("\nExample - Raw description (truncated 300 chars):\n", example_raw[:300])
print("\nExample - Cleaned description:\n", df['clean_description'].iloc[0])


# Part 3 — Feature Extraction (TF-IDF)

print("\n=== Part 3: Feature Extraction (TF-IDF) ===")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['clean_description'])
print("\nTF-IDF matrix shape:", X_tfidf.shape)

# 10 sample feature names
feature_names = tfidf.get_feature_names_out()
print("\n10 sample feature names:", feature_names[:10].tolist())

# Top 15 words by global TF-IDF (sum across docs)
tfidf_sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
top15_idx = np.argsort(tfidf_sums)[-15:][::-1]
top15 = [(feature_names[i], float(tfidf_sums[i])) for i in top15_idx]
print("\nTop 15 words by global TF-IDF score (word, summed_score):")
for w, s in top15:
    print(w, f"{s:.4f}")


# Part 4 — Model Building

print("\n=== Part 4: Model Building ===")
y = df['fraudulent'].astype(int)    # ensure int

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')  # balanced to help class imbalance
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nEvaluation on test set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nShort interpretation (2-3 sentences):")
print("The model achieves the metrics shown above. If precision is much higher than recall, the model is conservative (fewer false positives). If recall is higher, it catches more fakes but may include more false positives. Examine confusion matrix to judge trade-offs and consider more advanced models or feature engineering if needed.")

# === Part 4.1 — Model Comparison (Decision Tree & Random Forest) ===
print("\n=== Part 4.1: Model Comparison — Decision Tree & Random Forest ===")

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec = recall_score(y_te, preds, zero_division=0)
    f1s = f1_score(y_te, preds, zero_division=0)
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1s,
        "estimator": model
    }

results = []

# Baseline already trained Logistic Regression (reuse metrics computed above)
results.append({
    "model": "LogisticRegression",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "estimator": model
})

# Decision Trees: try depths 10, 20, 30
dt_depths = [10, 20, 30]
for d in dt_depths:
    dt = DecisionTreeClassifier(max_depth=d, class_weight='balanced', random_state=42)
    r = evaluate_model(f"DecisionTree(d={d})", dt, X_train, X_test, y_train, y_test)
    results.append(r)

# Random Forests: try n_estimators 50, 100, 200
rf_estimators = [50, 100, 200]
for n in rf_estimators:
    rf = RandomForestClassifier(
        n_estimators=n,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    r = evaluate_model(f"RandomForest(n={n})", rf, X_train, X_test, y_train, y_test)
    results.append(r)

results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False).reset_index(drop=True)
print("\nModel comparison (sorted by F1):")
print(results_df[["model", "accuracy", "precision", "recall", "f1"]])

# Pick best DT and best RF for plotting against LR
best_dt_row = results_df[results_df["model"].str.startswith("DecisionTree")].iloc[0]
best_rf_row = results_df[results_df["model"].str.startswith("RandomForest")].iloc[0]
best_dt_model = best_dt_row["estimator"]
best_rf_model = best_rf_row["estimator"]

joblib.dump(best_rf_model, 'fake_job_model_rf.pkl')
joblib.dump(best_dt_model, 'fake_job_model_dt.pkl')
print("Saved best RandomForest -> fake_job_model_rf.pkl")
print("Saved best DecisionTree -> fake_job_model_dt.pkl")

# Plot: F1-score comparison (LR vs best DT vs best RF)
labels = [
    "LogReg",
    best_dt_row["model"],
    best_rf_row["model"]
]
scores = [
    results_df[results_df["model"]=="LogisticRegression"].iloc[0]["f1"],
    best_dt_row["f1"],
    best_rf_row["f1"]
]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, scores, color=["#4e79a7", "#f28e2b", "#59a14f"])
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.title("Model Comparison (F1-score)")
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved model comparison chart to 'model_comparison.png'.")

# === Task 2 — Random Forest Feature Importance (top 15 TF-IDF terms) ===
print("\n=== Part 4.2: Random Forest Feature Importance (Top 15 words) ===")
feature_names = tfidf.get_feature_names_out()
rf_importances = best_rf_model.feature_importances_

top_idx = np.argsort(rf_importances)[-15:][::-1]
top_words = feature_names[top_idx]
top_scores = rf_importances[top_idx]

print("\nTop 15 important words (RF):")
for w, s in zip(top_words, top_scores):
    print(f"{w}: {s:.6f}")

# Plot horizontal bar chart
plt.figure(figsize=(8, 5))
ypos = np.arange(len(top_words))
plt.barh(ypos, top_scores[::-1], color="#59a14f")
plt.yticks(ypos, top_words[::-1])
plt.xlabel("Feature importance")
plt.title("Random Forest — Top 15 Important TF-IDF Words")
plt.tight_layout()
plt.savefig("rf_top15_features.png", dpi=150, bbox_inches="tight")
print("\nSaved RF top-features chart to 'rf_top15_features.png'.")

# === Part 4.3: Cross-Validation Analysis (5-fold) ===
print("\n=== Part 4.3: Cross-Validation (5-fold) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Densifier: named function instead of lambda (for pickling)
def densify(X):
    return X.toarray() if hasattr(X, "toarray") else X

to_dense = FunctionTransformer(densify, accept_sparse=True)

cv_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "DecisionTree": make_pipeline(to_dense, DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    "RandomForest": make_pipeline(to_dense, RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
}   

cv_stats = []
for name, clf in cv_models.items():
    scores = cross_val_score(clf, X_tfidf, y, cv=skf, scoring='accuracy', n_jobs=-1)
    mean_s, std_s = scores.mean(), scores.std()
    cv_stats.append((name, mean_s, std_s))
    print(f"{name} CV accuracy: mean={mean_s:.4f}, std={std_s:.4f}, scores={np.round(scores, 4)}")

# Plot mean CV accuracies
labels = [n for n, _, _ in cv_stats]
means = [m for _, m, _ in cv_stats]
plt.figure(figsize=(7, 4))
bars = plt.bar(labels, means, color=["#4e79a7", "#f28e2b", "#59a14f"])
plt.ylabel("Mean CV Accuracy (5-fold)")
plt.ylim(0, 1)
plt.title("5-fold CV Accuracy — LR vs DT vs RF")
for b, m in zip(bars, means):
    plt.text(b.get_x() + b.get_width()/2, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("cv_accuracy_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved CV accuracy comparison chart to 'cv_accuracy_comparison.png'.")

# Most stable model (lowest variance)
stable_name, _, stable_std = sorted(cv_stats, key=lambda t: t[2])[0]
print(f"Most stable (lowest std): {stable_name} (std={stable_std:.4f})")

# === Part 4.4: ROC-AUC Visualization ===
print("\n=== Part 4.4: ROC-AUC Visualization ===")
# Use already-fitted best/final models when available
lr_clf = model  # Logistic Regression already fitted
dt_clf = best_dt_model
rf_clf = best_rf_model

# Probabilities for ROC
lr_proba = lr_clf.predict_proba(X_test)[:, 1]
dt_proba = dt_clf.predict_proba(X_test)[:, 1]
rf_proba = rf_clf.predict_proba(X_test)[:, 1]

# Compute curves and AUC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

auc_lr = roc_auc_score(y_test, lr_proba)
auc_dt = roc_auc_score(y_test, dt_proba)
auc_rf = roc_auc_score(y_test, rf_proba)

print(f"AUC — Logistic: {auc_lr:.4f}, DecisionTree: {auc_dt:.4f}, RandomForest: {auc_rf:.4f}")
best_auc_model = max([("LogisticRegression", auc_lr), ("DecisionTree", auc_dt), ("RandomForest", auc_rf)], key=lambda x: x[1])
print(f"Best AUC: {best_auc_model[0]} ({best_auc_model[1]:.4f})")

plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic (AUC={auc_lr:.3f})", color="#4e79a7")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.3f})", color="#f28e2b")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})", color="#59a14f")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Logistic vs Decision Tree vs Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
print("Saved ROC curves to 'roc_curves.png'.")

# === Part 4.5: Hyperparameter Tuning — Decision Tree (Optional) ===
print("\n=== Part 4.5: Hyperparameter Tuning — Decision Tree (GridSearchCV) ===")
dt_pipe = make_pipeline(
    to_dense,
    DecisionTreeClassifier(class_weight='balanced', random_state=42)
)
param_grid = {
    "decisiontreeclassifier__max_depth": [10, 20, None],
    "decisiontreeclassifier__min_samples_split": [2, 10],
    "decisiontreeclassifier__criterion": ["gini"]
}
grid = GridSearchCV(
    estimator=dt_pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,  # reduce folds to 3
    n_jobs=-1,
    verbose=1
)
print("Fitting GridSearchCV with 3×2×1=6 candidates × 3 folds (faster)...")
grid.fit(X_train, y_train)
best_dt_tuned = grid.best_estimator_
print(f"Best DT params: {grid.best_params_}, CV mean acc={grid.best_score_:.4f}")

joblib.dump(best_dt_tuned, 'fake_job_model_dt_tuned.pkl')
print("Saved tuned DecisionTree -> fake_job_model_dt_tuned.pkl")

tuned_dt_acc = accuracy_score(y_test, best_dt_tuned.predict(X_test))
rf_acc = accuracy_score(y_test, best_rf_model.predict(X_test))
print(f"Tuned Decision Tree test accuracy: {tuned_dt_acc:.4f}")
print(f"Random Forest test accuracy:       {rf_acc:.4f}")
print("Reduced grid for speed. Full tuning can be done offline if needed.")

# Short guidance for interpretation (printed)
print("\nInterpretation tips:")
print("- Words related to scams (e.g., 'apply', 'fee', 'paypal', 'wire', 'urgent', 'limited', 'training', 'no experience') often rank higher for fake jobs.")
print("- Company/role-specific terms and detailed technical vocabulary tend to signal real posts.")
print("- Unexpected high-importance words may indicate data quirks or proxy signals; review samples where those words appear.")

# Part 5 — Model Analysis & Save

print("\n=== Part 5: Model Analysis & Saving ===")
# 5 random job descriptions from whole dataset (not only test); show predicted fake probability via pipeline
random_idx = random.sample(range(len(df)), 5)
sample_df = df.iloc[random_idx].copy()
X_sample_tfidf = tfidf.transform(sample_df['clean_description'])
sample_proba = model.predict_proba(X_sample_tfidf)[:, 1]
sample_pred_label = (sample_proba >= 0.5).astype(int)

sample_df = sample_df[['title', 'description', 'clean_description']].reset_index(drop=True)
sample_df['predicted_proba'] = sample_proba
sample_df['predicted_label'] = sample_pred_label

print("\n5 random job predictions (probability of being fake):\n")
print(sample_df[['title', 'predicted_proba', 'predicted_label']])

# Manually inspect one predicted fake and one predicted real (choose from sample_df)
fake_examples = sample_df[sample_df['predicted_label'] == 1]
real_examples = sample_df[sample_df['predicted_label'] == 0]

if not fake_examples.empty:
    print("\nExample predicted FAKE (inspect description):\n")
    print("Title:", fake_examples.iloc[0]['title'])
    print("Raw description (truncated):", fake_examples.iloc[0]['description'][:400])
    print("Cleaned:", fake_examples.iloc[0]['clean_description'][:400])
else:
    print("\nNo predicted-fake examples found in the random sample. You can rerun to sample again.")

if not real_examples.empty:
    print("\nExample predicted REAL (inspect description):\n")
    print("Title:", real_examples.iloc[0]['title'])
    print("Raw description (truncated):", real_examples.iloc[0]['description'][:400])
    print("Cleaned:", real_examples.iloc[0]['clean_description'][:400])
else:
    print("\nNo predicted-real examples found in the random sample. You can rerun to sample again.")

# Save trained model and vectorizer
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nSaved model to 'fake_job_model.pkl' and vectorizer to 'tfidf_vectorizer.pkl'")

print("\n=== Pipeline complete ===")
