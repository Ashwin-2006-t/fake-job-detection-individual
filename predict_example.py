import joblib

model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

samples = [
    "Work from home with high pay, no experience required! Apply immediately!",
    "We are hiring a software engineer with 3+ years of Python and Django experience."
]

X = vectorizer.transform(samples)
preds = model.predict(X)
prob = model.predict_proba(X)[:, 1]

for s, p, pr in zip(samples, preds, prob):
    label = "Fake Job" if p == 1 else "Real Job"
    print(f"\nText: {s}\nLabel: {label} | P(fake)={pr:.4f}")