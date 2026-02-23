"""
PhishGuard SVM - Flask API
===========================
REST API for phishing detection.

Setup:
  pip install flask flask-cors scikit-learn
  python train_model.py   # run once to create phishing_model.pkl
  python app.py

Endpoints:
  POST /predict   { "text": "email content here" }
  GET  /health
  GET  /features  (top discriminative words)
"""

import pickle
import json
import re
import os
from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

app = Flask(__name__)
if HAS_CORS:
    CORS(app)

# Load model
MODEL_PATH = 'phishing_model.pkl'
WEIGHTS_PATH = 'model_weights.json'

pipeline = None
feature_weights = {}

def load_model():
    global pipeline, feature_weights
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        print("✅ Model loaded")
    else:
        print("⚠️  Model not found. Run train_model.py first.")
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            feature_weights = json.load(f)

def extract_indicators(text):
    """Return human-readable indicators found in the message."""
    indicators = []
    t = text.lower()

    checks = [
        # (pattern, severity, message)
        (r'http[s]?://[^\s]*\.(xyz|info|tk|ml|ga|cf|gq|pw)[/\s]', 'high',
         '🔗 Suspicious domain TLD detected (.xyz, .info, etc.)'),
        (r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'high',
         '🔗 IP address used instead of domain name in URL'),
        (r'(urgent|act now|immediately|within 24 hours|expires today)', 'high',
         '⏰ High-urgency language to pressure action'),
        (r'(suspend|locked|terminated|closed|deactivat)', 'high',
         '🔒 Account suspension threat'),
        (r'(ssn|social security|credit card number|bank account|routing number)', 'high',
         '💳 Requesting sensitive financial credentials'),
        (r'(password|username|login credentials)', 'medium',
         '🔑 Requesting login credentials'),
        (r'(won|winner|prize|reward|lottery|congratulations.*free)', 'medium',
         '🎰 Prize or reward lure'),
        (r'(guaranteed|no risk|100% safe|risk.free).*(invest|return|profit)', 'medium',
         '📈 Unrealistic investment guarantee'),
        (r'(wire transfer|western union|bitcoin|crypto|gift card)', 'high',
         '💸 Unusual payment method requested'),
        (r'(arrest|legal action|lawsuit|prosecut|federal agent)', 'high',
         '⚖️ Legal threat to coerce compliance'),
        (r'verify.{0,30}(identity|account|information|details)', 'medium',
         '✅ Verification request pattern'),
        (r'(dear (customer|valued|user|member)|dear account holder)', 'low',
         '👤 Generic impersonal greeting'),
        (r'do not (share|tell|discuss|mention).{0,40}(confiden|secret)', 'high',
         '🤫 Secrecy request (BEC indicator)'),
    ]

    for pattern, severity, message in checks:
        if re.search(pattern, t):
            indicators.append({'severity': severity, 'text': message})

    return indicators


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': pipeline is not None})


@app.route('/features')
def features():
    return jsonify(feature_weights)


@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text provided'}), 400

    # SVM prediction
    prediction = int(pipeline.predict([text])[0])  # 0=legit, 1=phishing

    # Decision function gives distance from hyperplane (confidence proxy)
    decision_score = float(pipeline.decision_function([text])[0])

    # Normalize to 0-100 threat score using sigmoid-like mapping
    # Positive = phishing side, negative = legit side
    import math
    raw = max(-5, min(5, decision_score))           # clamp to [-5, 5]
    threat_score = int(50 + (raw / 5) * 50)         # map to [0, 100]
    threat_score = max(0, min(100, threat_score))

    # Verdict
    if threat_score >= 70:
        verdict = 'PHISHING'
        level = 'danger'
    elif threat_score >= 40:
        verdict = 'SUSPICIOUS'
        level = 'warn'
    else:
        verdict = 'LIKELY SAFE'
        level = 'safe'

    # Extract indicators
    indicators = extract_indicators(text)

    # Basic metadata
    urls = re.findall(r'http[s]?://\S+', text)
    sender_match = re.search(r'from:\s*(.+)', text, re.IGNORECASE)
    subject_match = re.search(r'subject:\s*(.+)', text, re.IGNORECASE)

    return jsonify({
        'verdict': verdict,
        'level': level,
        'threat_score': threat_score,
        'decision_score': round(decision_score, 4),
        'prediction': prediction,
        'indicators': indicators,
        'metadata': {
            'sender': sender_match.group(1).strip() if sender_match else 'Not found',
            'subject': subject_match.group(1).strip() if subject_match else 'Not found',
            'url_count': len(urls),
            'urls': urls[:5],
            'char_count': len(text),
            'word_count': len(text.split()),
        }
    })


if __name__ == '__main__':
    load_model()
    print("Starting PhishGuard API on http://localhost:5000")
    app.run(debug=True, port=5000)
