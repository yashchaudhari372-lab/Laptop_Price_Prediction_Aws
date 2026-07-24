import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the Decision Tree model from pickle
MODEL_PATH = "_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Single HTML template with CSS styling, JS theme switcher, and Chart.js integration
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Predictive Analytics Dashboard</title>
    <!-- Chart.js for visualization -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root[data-theme="dark"] {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.12);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --accent-primary: #8b5cf6;
            --accent-secondary: #ec4899;
            --accent-gradient: linear-gradient(135deg, #6366f1, #a855f7);
            --shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            --orb-opacity: 0.35;
        }

        :root[data-theme="light"] {
            --bg-gradient: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #cbd5e1 100%);
            --card-bg: rgba(255, 255, 255, 0.85);
            --card-border: rgba(0, 0, 0, 0.08);
            --text-main: #0f172a;
            --text-sub: #64748b;
            --input-bg: #ffffff;
            --input-border: rgba(0, 0, 0, 0.15);
            --accent-primary: #6366f1;
            --accent-secondary: #d946ef;
            --accent-gradient: linear-gradient(135deg, #4f46e5, #7c3aed);
            --shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
            --orb-opacity: 0.15;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Ambient Animated Orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(90px);
            opacity: var(--orb-opacity);
            z-index: 0;
            animation: float 12s ease-in-out infinite alternate;
            pointer-events: none;
        }
        .orb-1 { width: 350px; height: 350px; background: #6366f1; top: 5%; left: 10%; }
        .orb-2 { width: 300px; height: 300px; background: #ec4899; bottom: 5%; right: 10%; animation-delay: -6s; }

        @keyframes float {
            0% { transform: translateY(0px) scale(1); }
            100% { transform: translateY(-40px) scale(1.08); }
        }

        /* Navbar / Theme Toggle */
        .top-bar {
            position: absolute;
            top: 1.5rem;
            right: 2rem;
            z-index: 20;
        }

        .theme-btn {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.6rem 1.2rem;
            border-radius: 30px;
            cursor: pointer;
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .theme-btn:hover {
            transform: translateY(-2px);
        }

        /* Main Dashboard Grid Container */
        .dashboard-container {
            position: relative;
            z-index: 10;
            width: 100%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        @media (max-width: 900px) {
            .dashboard-container { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: var(--shadow);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card-header {
            margin-bottom: 1.5rem;
        }

        .card-header h2 {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-header p {
            color: var(--text-sub);
            font-size: 0.875rem;
            margin-top: 4px;
        }

        /* Input Form Elements */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
        }

        .full-width { grid-column: span 2; }

        @media (max-width: 500px) {
            .form-grid { grid-template-columns: 1fr; }
            .full-width { grid-column: span 1; }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-sub);
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-wrapper i {
            position: absolute;
            left: 14px;
            color: var(--text-sub);
        }

        .input-wrapper input, .input-wrapper select {
            width: 100%;
            padding: 0.75rem 0.75rem 0.75rem 2.5rem;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
        }

        .input-wrapper input:focus, .input-wrapper select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
        }

        /* Submit Button */
        .submit-btn {
            width: 100%;
            margin-top: 1.5rem;
            padding: 0.9rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            filter: brightness(1.1);
        }

        /* Output & Visualization Section */
        .result-badge {
            margin-bottom: 1.5rem;
            padding: 1.2rem;
            border-radius: 14px;
            text-align: center;
            animation: pulseIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .result-yes {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #22c55e;
        }

        .result-no {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #ef4444;
        }

        @keyframes pulseIn {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .chart-wrapper {
            position: relative;
            margin-top: 1rem;
            width: 100%;
            height: 260px;
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 280px;
            color: var(--text-sub);
            text-align: center;
        }

        .empty-state i {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
    </style>
</head>
<body>

    <!-- Light/Dark Mode Switcher -->
    <div class="top-bar">
        <button class="theme-btn" onclick="toggleTheme()">
            <i class="fa-solid fa-moon" id="theme-icon"></i>
            <span id="theme-text">Dark Mode</span>
        </button>
    </div>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="dashboard-container">
        <!-- Input Form Section -->
        <div class="card">
            <div class="card-header">
                <h2><i class="fa-solid fa-sliders" style="color: var(--accent-primary);"></i> Parameters</h2>
                <p>Provide customer details to analyze predictive outcome.</p>
            </div>

            <form action="/predict" method="POST">
                <div class="form-grid">
                    <div class="input-group">
                        <label for="age">Age</label>
                        <div class="input-wrapper">
                            <i class="fa-solid fa-user"></i>
                            <input type="number" id="age" name="Age" placeholder="e.g. 35" min="0" max="120" required value="{{ inputs.get('Age', '') }}">
                        </div>
                    </div>

                    <div class="input-group">
                        <label for="gender">Gender</label>
                        <div class="input-wrapper">
                            <i class="fa-solid fa-venus-mars"></i>
                            <select id="gender" name="Gender" required>
                                <option value="0" {% if inputs.get('Gender') == '0' %}selected{% endif %}>Female (0)</option>
                                <option value="1" {% if inputs.get('Gender') == '1' %}selected{% endif %}>Male (1)</option>
                            </select>
                        </div>
                    </div>

                    <div class="input-group">
                        <label for="region">Region Code</label>
                        <div class="input-wrapper">
                            <i class="fa-solid fa-location-dot"></i>
                            <input type="number" id="region" name="Region" placeholder="e.g. 1" required value="{{ inputs.get('Region', '') }}">
                        </div>
                    </div>

                    <div class="input-group">
                        <label for="occupation">Occupation</label>
                        <div class="input-wrapper">
                            <i class="fa-solid fa-briefcase"></i>
                            <input type="number" id="occupation" name="Occupation" placeholder="e.g. 2" required value="{{ inputs.get('Occupation', '') }}">
                        </div>
                    </div>

                    <div class="input-group full-width">
                        <label for="income">Income ($)</label>
                        <div class="input-wrapper">
                            <i class="fa-solid fa-wallet"></i>
                            <input type="number" step="any" id="income" name="Income" placeholder="e.g. 65000" required value="{{ inputs.get('Income', '') }}">
                        </div>
                    </div>
                </div>

                <button type="submit" class="submit-btn">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Execute Analysis
                </button>
            </form>
        </div>

        <!-- Result & Visual Chart Section -->
        <div class="card">
            <div class="card-header">
                <h2><i class="fa-solid fa-chart-pie" style="color: var(--accent-secondary);"></i> Output & Chart</h2>
                <p>Model prediction result and input feature metrics visualization.</p>
            </div>

            {% if result %}
                <div class="result-badge {{ 'result-yes' if result == 'yes' else 'result-no' }}">
                    <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">Classification Prediction</span>
                    <strong style="font-size: 1.8rem;">{{ result.upper() }}</strong>
                </div>

                <!-- Interactive Chart Container -->
                <div class="chart-wrapper">
                    <canvas id="featureChart"></canvas>
                </div>
            {% else %}
                <div class="empty-state">
                    <i class="fa-solid fa-chart-column"></i>
                    <p>Enter parameters on the left and click <strong>Execute Analysis</strong> to display model predictions and analytical breakdown.</p>
                </div>
            {% endif %}
        </div>
    </div>

    <!-- JavaScript logic for Theme Switch & Charting -->
    <script>
        // Theme Switcher Logic
        function toggleTheme() {
            const html = document.documentElement;
            const themeBtnText = document.getElementById('theme-text');
            const themeIcon = document.getElementById('theme-icon');
            
            if (html.getAttribute('data-theme') === 'dark') {
                html.setAttribute('data-theme', 'light');
                themeBtnText.textContent = 'Light Mode';
                themeIcon.className = 'fa-solid fa-sun';
                localStorage.setItem('theme', 'light');
            } else {
                html.setAttribute('data-theme', 'dark');
                themeBtnText.textContent = 'Dark Mode';
                themeIcon.className = 'fa-solid fa-moon';
                localStorage.setItem('theme', 'dark');
            }
        }

        // Restore saved theme on initial render
        (function() {
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-theme', savedTheme);
            window.addEventListener('DOMContentLoaded', () => {
                const themeBtnText = document.getElementById('theme-text');
                const themeIcon = document.getElementById('theme-icon');
                if (savedTheme === 'light' && themeBtnText && themeIcon) {
                    themeBtnText.textContent = 'Light Mode';
                    themeIcon.className = 'fa-solid fa-sun';
                }
            });
        })();

        // Chart Rendering (Executed when prediction results are available)
        {% if result %}
        const ctx = document.getElementById('featureChart').getContext('2d');
        const featureChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Age', 'Gender', 'Region', 'Occupation', 'Income (k$)'],
                datasets: [{
                    label: 'Normalized Parameter Inputs',
                    data: [
                        {{ inputs.get('Age', 0) }},
                        {{ inputs.get('Gender', 0) }},
                        {{ inputs.get('Region', 0) }},
                        {{ inputs.get('Occupation', 0) }},
                        {{ (inputs.get('Income', 0) | float) / 1000 }} // Scaled for chart readability
                    ],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.7)',
                        'rgba(168, 85, 247, 0.7)',
                        'rgba(236, 72, 153, 0.7)',
                        'rgba(34, 197, 94, 0.7)',
                        'rgba(245, 158, 11, 0.7)'
                    ],
                    borderColor: [
                        '#6366f1',
                        '#a855f7',
                        '#ec4899',
                        '#22c55e',
                        '#f59e0b'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
        {% endif %}
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, result=None, inputs={})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model standard loading error. Ensure _model.pkl is in the same directory.", 500

    try:
        # Extract inputs from form
        form_data = request.form.to_dict()
        
        # Prepare DataFrame strictly matching feature names extracted from the pickle binary
        input_data = pd.DataFrame([{
            'Age': float(form_data.get('Age', 0)),
            'Gender': float(form_data.get('Gender', 0)),
            'Region': float(form_data.get('Region', 0)),
            'Occupation': float(form_data.get('Occupation', 0)),
            'Income': float(form_data.get('Income', 0))
        }])

        # Perform classification prediction
        prediction = model.predict(input_data)[0]

        return render_template_string(HTML_TEMPLATE, result=str(prediction), inputs=form_data)

    except Exception as e:
        return f"Prediction Error: {str(e)}", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
