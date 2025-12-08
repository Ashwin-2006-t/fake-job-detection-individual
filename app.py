from flask import Flask, render_template, request, redirect, url_for, session
import joblib, sqlite3
from datetime import datetime

DB_PATH = 'job_predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        );
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS retrain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            accuracy REAL,
            training_source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    cur = conn.execute("SELECT COUNT(*) FROM admin WHERE username='admin'")
    if cur.fetchone()[0] == 0:
        conn.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ('admin', 'admin123'))
    conn.commit()
    conn.close()

init_db()

app = Flask(__name__)
app.secret_key = "mysecretkey123"

model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Force login before showing index
def get_counts():
    conn = sqlite3.connect(DB_PATH)
    fake_jobs = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    real_jobs = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    conn.close()
    return fake_jobs, real_jobs

# ...existing code...
@app.route('/')
def home():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    fake_jobs, real_jobs = get_counts()
    
    # Fetch last prediction
    conn = sqlite3.connect(DB_PATH)
    last_row = conn.execute('SELECT job_description, prediction, confidence FROM predictions ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    
    last_text = last_row[0] if last_row else None
    last_label = last_row[1] if last_row else None
    last_confidence = last_row[2] if last_row else None
    
    return render_template('index.html', fake=fake_jobs, real=real_jobs,
                           last_text=last_text, last_label=last_label, last_confidence=last_confidence)
# ...existing code...

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    job_desc = request.form.get('job_description','').strip()
    fake_jobs, real_jobs = get_counts()  # current counts for error re-render
    if not job_desc or len(job_desc.split()) < 5:
        return render_template('index.html', error="Please enter ≥5 words.", fake=fake_jobs, real=real_jobs)
    letters = sum(c.isalpha() for c in job_desc)
    if letters / max(1,len(job_desc)) < 0.40:
        return render_template('index.html', error="Too many symbols/numbers.", fake=fake_jobs, real=real_jobs)

    X = vectorizer.transform([job_desc])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob*100,2) if pred==1 else round((1-prob)*100,2)

    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',
                 (job_desc, label, confidence))
    conn.commit()
    conn.close()

    # updated counts after insert
    fake_jobs, real_jobs = get_counts()
    return render_template('result.html', label=label, confidence=confidence, description=job_desc,
                           fake=fake_jobs, real=real_jobs)

@app.route('/history')
def history():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT timestamp, job_description, prediction, confidence FROM predictions ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('history.html', records=[(r[1], r[2], r[3], r[0]) for r in rows])  # match job,label,conf,time

@app.route('/admin_login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username','')
        password = request.form.get('password','')
        conn = sqlite3.connect(DB_PATH)
        admin = conn.execute("SELECT id FROM admin WHERE username=? AND password=?", (username,password)).fetchone()
        conn.close()
        if admin:
            session['admin_logged_in'] = True
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    fake_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    real_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    total = fake_count + real_count
    
    # Daily count with proper date formatting
    daily_data = cursor.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as cnt
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """).fetchall()
    
    # Ensure at least 2 points for line chart (pad with zero if needed)
    if len(daily_data) == 0:
        dates, counts = [], []
    elif len(daily_data) == 1:
        # Add a dummy previous day with 0
        from datetime import datetime, timedelta
        single_date = datetime.strptime(daily_data[0][0], '%Y-%m-%d')
        prev_date = (single_date - timedelta(days=1)).strftime('%Y-%m-%d')
        dates = [prev_date, daily_data[0][0]]
        counts = [0, daily_data[0][1]]
    else:
        dates = [row[0] for row in daily_data]
        counts = [row[1] for row in daily_data]
    
    conn.close()
    
    return render_template('dashboard.html',
                           total=total,
                           fake=fake_count,
                           real=real_count,
                           dates=dates,
                           counts=counts)

@app.route('/retrain', methods=['POST'])
def retrain():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    # SIMULATED retraining accuracy (replace later if needed)
    accuracy = 95.2
    training_source = "default dataset"

    # Save record in retrain_logs
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO retrain_logs (accuracy, training_source) VALUES (?, ?)",
                 (accuracy, training_source))
    conn.commit()
    conn.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/retrain_logs')
def retrain_logs():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT timestamp, accuracy, training_source FROM retrain_logs ORDER BY id DESC").fetchall()
    conn.close()

    return render_template('retrain_logs.html', logs=rows)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    app.run(debug=True)