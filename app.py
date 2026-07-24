import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = "_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Embedded HTML Template with full CSS and Animated UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Dashboard</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.07);
            --glass-border: rgba(255, 255, 255, 0.15);
            --accent-glow: #8b5cf6;
            --accent-hover: #7c3aed;
            --text-color: #f8fafc;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-color);
            overflow-x: hidden;
            position: relative;
        }

        /* Background Animated Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.5;
            z-index: 0;
            animation: float 10s ease-in-out infinite alternate;
        }

        .orb-1 {
            width: 300px;
            height: 300px;
            background: #6366f1;
            top: 10%;
            left: 15%;
        }

        .orb-2 {
            width: 250px;
            height: 250px;
            background: #ec4899;
            bottom: 10%;
            right: 15%;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translateY(0px) scale(1); }
            100% { transform: translateY(-30px) scale(1.1); }
        }

        /* Main Container */
        .card {
            position: relative;
            z-index: 10;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            width: 100%;
            max-width: 480px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: cardAppear 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes cardAppear {
            from { opacity: 0; transform: translateY(20px) scale(0.95); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        h2 {
            text-align: center;
            font-weight: 700;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 1.8rem;
        }

        .form-group {
            margin-bottom: 1.2rem;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            width: 100%;
            padding: 0.8rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);
        }

        select option {
            background: #1e1b4b;
            color: #fff;
        }

        button {
            width: 100%;
            padding: 0.9rem;
            margin-top: 1rem;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border: none;
            border-radius: 12px;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            transition: all 0.3s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
        }

        /* Result Animation Box */
        .result-box {
            margin-top: 1.8rem;
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            animation: pulseResult 0.6s ease-out;
        }

        .result-yes {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }

        .result-no {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }

        @keyframes pulseResult {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="card">
        <h2>Prediction Portal</h2>
        <p class="subtitle">Enter details to run decision tree model</p>

        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="Age" placeholder="e.g. 25" required value="{{ inputs.get('Age', '') }}">
            </div>

            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="Gender" required>
                    <option value="0" {% if inputs.get('Gender') == '0' %}selected{% endif %}>Female (0)</option>
                    <option value="1" {% if inputs.get('Gender') == '1' %}selected{% endif %}>Male (1)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="region">Region</label>
                <input type="number" id="region" name="Region" placeholder="Region code (e.g. 1)" required value="{{ inputs.get('Region', '') }}">
            </div>

            <div class="form-group">
                <label for="occupation">Occupation</label>
                <input type="number" id="occupation" name="Occupation" placeholder="Occupation code (e.g. 2)" required value="{{ inputs.get('Occupation', '') }}">
            </div>

            <div class="form-group">
                <label for="income">Income</label>
                <input type="number" step="any" id="income" name="Income" placeholder="e.g. 50000" required value="{{ inputs.get('Income', '') }}">
            </div>

            <button type="submit">Predict Outcome</button>
        </form>

        {% if result %}
            <div class="result-box {{ 'result-yes' if result == 'yes' else 'result-no' }}">
                Prediction Result: <strong>{{ result.upper() }}</strong>
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, result=None, inputs={})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model file missing or corrupted.", 500

    try:
        # Get raw form data
        form_data = request.form.to_dict()
        
        # Prepare DataFrame matching exact pickle feature names
        input_data = pd.DataFrame([{
            'Age': float(form_data.get('Age', 0)),
            'Gender': float(form_data.get('Gender', 0)),
            'Region': float(form_data.get('Region', 0)),
            'Occupation': float(form_data.get('Occupation', 0)),
            'Income': float(form_data.get('Income', 0))
        }])

        # Perform prediction
        prediction = model.predict(input_data)[0]

        return render_template_string(HTML_TEMPLATE, result=str(prediction), inputs=form_data)

    except Exception as e:
        return f"Prediction Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
