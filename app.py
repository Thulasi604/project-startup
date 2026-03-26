from flask import Flask, request, jsonify, render_template_string, make_response
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import traceback
import warnings
from pathlib import Path

from xgboost import XGBClassifier

# Compatibility import for older pickled model artifacts that reference setuptools.
try:
    import pkg_resources  # noqa: F401
except Exception:
    pkg_resources = None

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

DEFAULT_CITY = 3522  # San Francisco

# -------------------------------------------------------------------
# Model loading (with compatibility fixes)
# -------------------------------------------------------------------
model = None
model_path = None

def clean_xgboost_model(model_obj):
    for attr in ['use_label_encoder', 'gpu_id']:
        if hasattr(model_obj, attr):
            delattr(model_obj, attr)
            print(f"   Removed '{attr}' attribute")
    return model_obj

model_candidates = [
    Path("startup_model.pkl"),
    Path("startup_model.json"),
]

for candidate in model_candidates:
    if candidate.exists():
        model_path = candidate
        break

if model_path is not None:
    try:
        if model_path.suffix == ".json":
            model = XGBClassifier()
            model.load_model(model_path)
            print(f"[OK] XGBoost JSON model loaded from '{model_path.name}'")
        else:
            with model_path.open("rb") as f:
                raw_model = pickle.load(f)
            print(f"[OK] Raw model loaded: type={type(raw_model)}")
            model = clean_xgboost_model(raw_model)

        print("[OK] Model ready for predictions")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        traceback.print_exc()
        model = None
else:
    print("[ERROR] No supported model file found (expected startup_model.pkl or startup_model.json).")

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Rule-based fallback prediction (used when model fails or isn't loaded)
# -------------------------------------------------------------------
def rule_based_predict(data):
    """
    Calculate a success score based on business rules.
    Returns (prediction, score) where:
    - prediction: 1 for success, 0 for risk
    - score: 0-100 score
    """
    score = 0
    
    # 1. Funding Amount (max 30 points)
    funding = data.get('funding_total_usd', 0)
    if funding >= 10000000:      # $10M+
        score += 30
    elif funding >= 5000000:     # $5M - $10M
        score += 25
    elif funding >= 2000000:     # $2M - $5M
        score += 20
    elif funding >= 1000000:     # $1M - $2M
        score += 15
    elif funding >= 500000:      # $500K - $1M
        score += 10
    else:                         # < $500K
        score += 5
    
    # 2. Funding Rounds (max 20 points)
    rounds = data.get('funding_rounds', 1)
    if rounds >= 5:
        score += 20
    elif rounds >= 4:
        score += 18
    elif rounds >= 3:
        score += 15
    elif rounds >= 2:
        score += 10
    else:
        score += 5
    
    # 3. Startup Age (max 20 points)
    age = data.get('startup_age', 0)
    if age >= 8:
        score += 20
    elif age >= 5:
        score += 15
    elif age >= 3:
        score += 10
    elif age >= 1:
        score += 5
    else:
        score += 0
    
    # 4. Funding Duration (max 15 points)
    duration = data.get('funding_duration', 0)
    if duration >= 5:
        score += 15
    elif duration >= 3:
        score += 12
    elif duration >= 2:
        score += 8
    elif duration >= 1:
        score += 5
    else:
        score += 0
    
    # 5. Industry (max 10 points)
    industry = data.get('category_list', 3090)
    # High success industries (Software, Biotech, Enterprise Software)
    if industry in [3988, 3598, 3980]:
        score += 10
    # Medium success industries (E-Commerce, Mobile, Healthcare)
    elif industry in [1328, 1175, 3704]:
        score += 7
    # Low success industries
    else:
        score += 4
    
    # 6. Location (max 5 points)
    country = data.get('country_code', 6933)
    if country == 37242:  # USA
        score += 5
    elif country in [3668, 1909]:  # UK, Canada
        score += 4
    elif country in [1586, 1544, 975]:  # India, China, Germany
        score += 3
    else:
        score += 2
    
    # Determine prediction based on score
    # Threshold: 50 points = 50% probability
    prediction = 1 if score >= 50 else 0
    
    return prediction, score

# -------------------------------------------------------------------
# Home route – HTML embedded as raw string
# -------------------------------------------------------------------
@app.route('/')
def home():
    # IMPORTANT: Paste your original, complete HTML string here (the one with all sliders and dropdowns).
    # The HTML must contain the following field IDs: fundingSlider, roundsSlider, ageSlider,
    # durationSlider, industrySelect, countrySelect, and the result placeholders.
    # It must NOT have an extra city dropdown (unless you also adjust the prediction logic).
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Startup Success Predictor | Data-Driven Analytics</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            margin-bottom: 40px;
            display: flex;
            align-items: center;
            gap: 30px;
            flex-wrap: wrap;
        }

        .header-content {
            flex: 1;
        }

        .header-image {
            flex: 0 0 300px;
            text-align: center;
        }

        .header-image img {
            max-width: 100%;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }

        .header h1 {
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            max-width: 600px;
            line-height: 1.6;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 25px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .stat-card:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }

        .stat-icon {
            width: 60px;
            height: 60px;
            background: rgba(255,255,255,0.2);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: white;
        }

        .stat-info {
            flex: 1;
        }

        .stat-label {
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }

        .stat-value {
            color: white;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .stat-desc {
            color: rgba(255,255,255,0.6);
            font-size: 0.8rem;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 30px;
            margin-bottom: 50px;
        }

        .form-card {
            background: white;
            border-radius: 24px;
            padding: 35px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }

        .form-card h2 {
            color: #1e293b;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .form-card h2 i {
            color: #667eea;
            font-size: 2rem;
        }

        .form-card .subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 30px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        .form-group {
            margin-bottom: 5px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        .form-group label {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #475569;
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 8px;
        }

        .form-group label i {
            color: #667eea;
            width: 20px;
            font-size: 1.1rem;
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .slider {
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            outline: none;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            border: 2px solid #667eea;
            transition: all 0.2s;
        }

        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            background: #667eea;
        }

        .slider-value {
            min-width: 80px;
            padding: 6px 12px;
            background: #f1f5f9;
            border-radius: 8px;
            font-weight: 600;
            color: #1e293b;
            text-align: center;
            border: 1px solid #e2e8f0;
        }

        .slider-value i {
            margin-right: 4px;
            color: #667eea;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            font-size: 0.95rem;
            transition: all 0.2s;
            background: #f8fafc;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }

        .input-hint {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 5px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .input-hint i {
            color: #667eea;
            font-size: 0.75rem;
        }

        .btn-primary {
            grid-column: span 2;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 16px;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102,126,234,0.4);
        }

        .btn-primary i {
            font-size: 1.2rem;
        }

        .result-card {
            background: white;
            border-radius: 24px;
            padding: 35px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }

        .result-card h2 {
            color: #1e293b;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .result-card h2 i {
            color: #764ba2;
            font-size: 2rem;
        }

        .result-card .subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 30px;
        }

        .result-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 30px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 20px;
        }

        .result-badge.success {
            background: #d1fae5;
            color: #065f46;
        }

        .result-badge.warning {
            background: #fee2e2;
            color: #991b1b;
        }

        .result-image {
            text-align: center;
            margin: 20px 0;
        }

        .result-image img {
            max-width: 200px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }

        .result-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 15px;
            text-align: center;
        }

        .result-message {
            color: #475569;
            line-height: 1.6;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }

        .metrics-section {
            border-top: 1px solid #e2e8f0;
            padding-top: 25px;
        }

        .metric-item {
            margin-bottom: 20px;
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            color: #475569;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .metric-bar {
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .metric-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .no-result {
            color: #94a3b8;
            text-align: center;
            padding: 60px 20px;
        }

        .no-result i {
            font-size: 4rem;
            margin-bottom: 20px;
            opacity: 0.5;
            color: #667eea;
        }

        .no-result img {
            max-width: 200px;
            margin-bottom: 20px;
            opacity: 0.7;
        }

        .factors-section {
            margin-top: 50px;
        }

        .factors-section h2 {
            color: white;
            font-size: 2.2rem;
            font-weight: 600;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .factors-section h2 i {
            font-size: 2rem;
            color: #ffd700;
        }

        .factors-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 25px;
        }

        .factor-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }

        .factor-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }

        .factor-card:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.3);
            transform: translateY(-5px);
        }

        .factor-icon {
            width: 60px;
            height: 60px;
            background: rgba(255,255,255,0.2);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 2rem;
            margin-bottom: 20px;
        }

        .factor-title {
            color: white;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .factor-desc {
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 15px;
        }

        .factor-tip {
            color: rgba(255,255,255,0.9);
            font-size: 0.85rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }

        .factor-tip i {
            color: #ffd700;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }

        .spinner {
            border: 3px solid rgba(255,255,255,0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error-message {
            background: #fee2e2;
            color: #991b1b;
            padding: 15px;
            border-radius: 12px;
            margin-top: 20px;
            display: none;
            border: 1px solid #fecaca;
        }

        .footer {
            margin-top: 60px;
            text-align: center;
            color: rgba(255,255,255,0.6);
            font-size: 0.9rem;
            padding: 20px 0;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        @media (max-width: 1024px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .factors-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .header {
                flex-direction: column;
                text-align: center;
            }
            .header-image {
                flex: 0 0 auto;
            }
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            .form-grid {
                grid-template-columns: 1fr;
            }
            .btn-primary {
                grid-column: span 1;
            }
            .factors-grid {
                grid-template-columns: 1fr;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>🚀 Startup Success Predictor</h1>
                <p>Data-driven insights based on analysis of 65,871 global startups across 138 countries. Use the interactive sliders to explore different scenarios.</p>
            </div>
            <div class="header-image">
                <img src="https://img.freepik.com/free-vector/startup-concept-illustration_114360-2341.jpg" alt="Startup Illustration" onerror="this.src='https://via.placeholder.com/300x200?text=Startup+Analytics'">
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-building"></i></div>
                <div class="stat-info">
                    <div class="stat-label">Total Startups</div>
                    <div class="stat-value">65,871</div>
                    <div class="stat-desc">Comprehensive dataset</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                <div class="stat-info">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-value">89.5%</div>
                    <div class="stat-desc">Operating / Acquired</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-globe"></i></div>
                <div class="stat-info">
                    <div class="stat-label">Countries</div>
                    <div class="stat-value">138</div>
                    <div class="stat-desc">Global coverage</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-industry"></i></div>
                <div class="stat-info">
                    <div class="stat-label">Industries</div>
                    <div class="stat-value">107</div>
                    <div class="stat-desc">Sectors analyzed</div>
                </div>
            </div>
        </div>

        <div class="main-grid">
            <div class="form-card">
                <h2><i class="fas fa-sliders-h"></i> Interactive Startup Profile</h2>
                <div class="subtitle">Adjust the sliders to see real-time impact on your success probability</div>
                
                <form id="predictionForm">
                    <div class="form-grid">
                        <div class="form-group full-width">
                            <label>
                                <i class="fas fa-coins"></i>
                                Funding Amount: <span id="fundingAmountDisplay" class="slider-value"><i class="fas fa-dollar-sign"></i>5,000,000</span>
                            </label>
                            <div class="slider-container">
                                <input type="range" id="fundingSlider" class="slider" min="0" max="20000000" step="100000" value="5000000">
                                <input type="hidden" name="funding_total_usd" id="fundingInput" value="5000000">
                            </div>
                            <div class="input-hint">
                                <i class="fas fa-info-circle"></i>
                                Drag to adjust: <span id="fundingHint">$5M (Good)</span>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label>
                                <i class="fas fa-layer-group"></i>
                                Funding Rounds: <span id="roundsDisplay" class="slider-value">3</span>
                            </label>
                            <div class="slider-container">
                                <input type="range" id="roundsSlider" class="slider" min="1" max="10" step="1" value="3">
                                <input type="hidden" name="funding_rounds" id="roundsInput" value="3">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label>
                                <i class="fas fa-calendar"></i>
                                Startup Age: <span id="ageDisplay" class="slider-value">3.0</span>
                            </label>
                            <div class="slider-container">
                                <input type="range" id="ageSlider" class="slider" min="0" max="15" step="0.5" value="3">
                                <input type="hidden" name="startup_age" id="ageInput" value="3">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label>
                                <i class="fas fa-hourglass-half"></i>
                                Funding Duration: <span id="durationDisplay" class="slider-value">2.0</span>
                            </label>
                            <div class="slider-container">
                                <input type="range" id="durationSlider" class="slider" min="0" max="10" step="0.5" value="2">
                                <input type="hidden" name="funding_duration" id="durationInput" value="2">
                            </div>
                        </div>
                        
                        <div class="form-group full-width">
                            <label>
                                <i class="fas fa-chart-pie"></i>
                                Industry
                            </label>
                            <select name="category_list" id="industrySelect" required>
                                <option value="3988" selected>💻 Software & Technology</option>
                                <option value="3598">🧬 Biotechnology</option>
                                <option value="1328">🛍️ E-Commerce</option>
                                <option value="1175">📱 Mobile Applications</option>
                                <option value="3980">🏢 Enterprise Software</option>
                                <option value="3704">🏥 Healthcare</option>
                                <option value="3590">🌱 Clean Technology</option>
                                <option value="3421">📢 Advertising & Marketing</option>
                                <option value="3090">❓ Other / Not Specified</option>
                            </select>
                        </div>
                        
                        <div class="form-group full-width">
                            <label>
                                <i class="fas fa-map-marker-alt"></i>
                                Country
                            </label>
                            <select name="country_code" id="countrySelect" required>
                                <option value="37242" selected>🇺🇸 United States</option>
                                <option value="3668">🇬🇧 United Kingdom</option>
                                <option value="1909">🇨🇦 Canada</option>
                                <option value="1586">🇮🇳 India</option>
                                <option value="1544">🇨🇳 China</option>
                                <option value="975">🇩🇪 Germany</option>
                                <option value="866">🇫🇷 France</option>
                                <option value="698">🇦🇺 Australia</option>
                                <option value="572">🇮🇱 Israel</option>
                                <option value="6933">🌍 Other / Unknown</option>
                            </select>
                        </div>
                        
                        <button type="submit" class="btn-primary">
                            <i class="fas fa-magic"></i>
                            Analyze My Startup
                        </button>
                    </div>
                </form>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="color: #64748b;">Analyzing your startup data...</p>
                </div>

                <div class="error-message" id="errorMessage"></div>
            </div>

            <div class="result-card">
                <h2><i class="fas fa-chart-bar"></i> Analysis Results</h2>
                <div class="subtitle">Your personalized startup success analysis</div>

                <div id="result" style="display: none;">
                    <div id="resultBadge" class="result-badge"></div>
                    
                    <div class="result-image" id="resultImage"></div>
                    
                    <div id="resultValue" class="result-value"></div>
                    <div id="resultMessage" class="result-message"></div>

                    <div class="metrics-section">
                        <div class="metric-item">
                            <div class="metric-header">
                                <span><i class="fas fa-coins" style="color: #667eea;"></i> Funding Health</span>
                                <span id="fundingScore">0%</span>
                            </div>
                            <div class="metric-bar">
                                <div class="metric-fill" id="fundingBar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-header">
                                <span><i class="fas fa-chart-line" style="color: #667eea;"></i> Market Position</span>
                                <span id="marketScore">0%</span>
                            </div>
                            <div class="metric-bar">
                                <div class="metric-fill" id="marketBar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-header">
                                <span><i class="fas fa-map-marker-alt" style="color: #667eea;"></i> Location Advantage</span>
                                <span id="locationScore">0%</span>
                            </div>
                            <div class="metric-bar">
                                <div class="metric-fill" id="locationBar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="noResult" class="no-result">
                    <i class="fas fa-chart-line"></i>
                    <img src="https://img.freepik.com/free-vector/business-analysis-concept-illustration_114360-1233.jpg" alt="Analysis" onerror="this.style.display='none'">
                    <p>Adjust the sliders and click "Analyze" to see your results</p>
                </div>
            </div>
        </div>

        <div class="factors-section">
            <h2><i class="fas fa-exclamation-triangle"></i> Critical Success Factors</h2>
            <div class="factors-grid">
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-coins"></i></div>
                    <div class="factor-title">Funding Amount</div>
                    <div class="factor-desc">Startups with >$5M funding show 40% higher survival rates.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> Target: $2M - $10M</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-layer-group"></i></div>
                    <div class="factor-title">Funding Rounds</div>
                    <div class="factor-desc">3-5 rounds indicate sustained investor confidence.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> Aim for 3+ rounds</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-clock"></i></div>
                    <div class="factor-title">Startup Age</div>
                    <div class="factor-desc">Survival rate increases 15% annually after year 3.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> Survive first 3 years</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-calendar-check"></i></div>
                    <div class="factor-title">Funding Duration</div>
                    <div class="factor-desc">2-4 years of consistent funding demonstrates sustainability.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> 18-24 month runway</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-chart-bar"></i></div>
                    <div class="factor-title">Industry Selection</div>
                    <div class="factor-desc">Software leads with 92% success rate.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> Software: 92% success</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-globe"></i></div>
                    <div class="factor-title">Location</div>
                    <div class="factor-desc">US, UK, and Canada offer strongest ecosystems.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> Access to capital & talent</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-city"></i></div>
                    <div class="factor-title">City Selection</div>
                    <div class="factor-desc">SF, NYC, London show 2x success rates.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> SF: 2x success rate</div>
                </div>
                <div class="factor-card">
                    <div class="factor-icon"><i class="fas fa-rocket"></i></div>
                    <div class="factor-title">Funding Trajectory</div>
                    <div class="factor-desc">2-3x round growth signals market validation.</div>
                    <div class="factor-tip"><i class="fas fa-bullseye"></i> 2-3x round increases</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>© 2026 Startup Success Predictor | Data updated March 2026</p>
            <p style="margin-top: 10px; font-size: 0.8rem;">Images by Freepik</p>
        </div>
    </div>

    <script>
        const fundingSlider = document.getElementById('fundingSlider');
        const fundingDisplay = document.getElementById('fundingAmountDisplay');
        const fundingInput = document.getElementById('fundingInput');
        const fundingHint = document.getElementById('fundingHint');
        
        const roundsSlider = document.getElementById('roundsSlider');
        const roundsDisplay = document.getElementById('roundsDisplay');
        const roundsInput = document.getElementById('roundsInput');
        
        const ageSlider = document.getElementById('ageSlider');
        const ageDisplay = document.getElementById('ageDisplay');
        const ageInput = document.getElementById('ageInput');
        
        const durationSlider = document.getElementById('durationSlider');
        const durationDisplay = document.getElementById('durationDisplay');
        const durationInput = document.getElementById('durationInput');

        function formatCurrency(value) {
            if (value >= 1000000) return '$' + (value / 1000000).toFixed(1) + 'M';
            if (value >= 1000) return '$' + (value / 1000).toFixed(0) + 'K';
            return '$' + value;
        }

        function getFundingHint(value) {
            if (value >= 10000000) return 'Excellent (>$10M)';
            if (value >= 5000000) return 'Good ($5M-$10M)';
            if (value >= 2000000) return 'Moderate ($2M-$5M)';
            if (value >= 1000000) return 'Adequate ($1M-$2M)';
            if (value >= 500000) return 'Limited ($500K-$1M)';
            return 'Low (<$500K)';
        }

        fundingSlider.addEventListener('input', function() {
            const value = parseInt(this.value);
            fundingDisplay.innerHTML = `<i class="fas fa-dollar-sign"></i> ${value.toLocaleString()}`;
            fundingInput.value = value;
            fundingHint.textContent = getFundingHint(value);
        });

        roundsSlider.addEventListener('input', function() {
            roundsDisplay.textContent = this.value;
            roundsInput.value = this.value;
        });

        ageSlider.addEventListener('input', function() {
            ageDisplay.textContent = parseFloat(this.value).toFixed(1);
            ageInput.value = this.value;
        });

        durationSlider.addEventListener('input', function() {
            durationDisplay.textContent = parseFloat(this.value).toFixed(1);
            durationInput.value = this.value;
        });

        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('noResult').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            
            const formData = {
                category_list: parseFloat(document.getElementById('industrySelect').value),
                funding_total_usd: parseFloat(document.getElementById('fundingInput').value),
                country_code: parseFloat(document.getElementById('countrySelect').value),
                funding_rounds: parseFloat(document.getElementById('roundsInput').value),
                startup_age: parseFloat(document.getElementById('ageInput').value),
                funding_duration: parseFloat(document.getElementById('durationInput').value)
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    document.getElementById('result').style.display = 'block';
                    
                    document.getElementById('fundingScore').textContent = Math.round(data.funding_score) + '%';
                    document.getElementById('marketScore').textContent = Math.round(data.market_score) + '%';
                    document.getElementById('locationScore').textContent = Math.round(data.location_score) + '%';
                    
                    document.getElementById('fundingBar').style.width = data.funding_score + '%';
                    document.getElementById('marketBar').style.width = data.market_score + '%';
                    document.getElementById('locationBar').style.width = data.location_score + '%';
                    
                    const badge = document.getElementById('resultBadge');
                    const value = document.getElementById('resultValue');
                    const message = document.getElementById('resultMessage');
                    
                    if (data.prediction == 1) {
                        badge.className = 'result-badge success';
                        badge.textContent = 'HIGH POTENTIAL';
                        value.textContent = 'Success Likely';
                        value.style.color = '#065f46';
                        message.textContent = data.message;
                    } else {
                        badge.className = 'result-badge warning';
                        badge.textContent = 'ELEVATED RISK';
                        value.textContent = 'Caution Advised';
                        value.style.color = '#991b1b';
                        message.textContent = data.message;
                    }
                } else {
                    document.getElementById('errorMessage').textContent = 'Error: ' + data.error;
                    document.getElementById('errorMessage').style.display = 'block';
                    document.getElementById('noResult').style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('errorMessage').textContent = 'Connection error. Please try again.';
                document.getElementById('errorMessage').style.display = 'block';
                document.getElementById('noResult').style.display = 'block';
            });
        });
    </script>
</body>
</html>'''
    response = make_response(render_template_string(html))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

# -------------------------------------------------------------------
# Prediction endpoint – uses trained XGBoost model if available, else fallback
# -------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        data['city'] = DEFAULT_CITY

        if model is not None:
            try:
                # Build DataFrame with the exact column order the model expects
                input_df = pd.DataFrame([[
                    data['category_list'],
                    data['funding_total_usd'],
                    data['country_code'],
                    data['city'],
                    data['funding_rounds'],
                    data['startup_age'],
                    data['funding_duration']
                ]], columns=['category_list', 'funding_total_usd', 'country_code', 'city',
                             'funding_rounds', 'startup_age', 'funding_duration'])

                # Get probability of success (class 1)
                proba = model.predict_proba(input_df)[0][1]   # NumPy float
                score = round(float(proba) * 100, 2)          # 0‑100 score
                prediction = 1 if proba >= 0.5 else 0
                message = f"Model predicts {'SUCCESS' if prediction == 1 else 'RISK'} with {score}% confidence."

            except Exception as e:
                print(f"Model prediction failed: {e}. Using rule‑based fallback.")
                prediction, score = rule_based_predict(data)
                score = int(score)
                message = f"Rule‑based analysis (score {score}/100): {'Success likely' if prediction == 1 else 'High risk'}."
        else:
            prediction, score = rule_based_predict(data)
            score = int(score)
            message = f"Rule‑based analysis (score {score}/100): {'Success likely' if prediction == 1 else 'High risk'}."

        # Compute the three metrics (same as before)
        funding_score = min(100, (data['funding_total_usd'] / 10_000_000) * 100)
        market_score = min(100, (data['startup_age'] * 10) + (data['funding_duration'] * 10))
        location_score = 90 if data['country_code'] == 37242 else 80 if data['country_code'] in [3668, 1909] else 70

        # Convert all to Python native types for JSON serialization
        funding_score = float(funding_score)
        market_score = float(market_score)
        location_score = float(location_score)
        prediction = int(prediction)
        score = float(score)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'score': score,
            'message': message,
            'funding_score': funding_score,
            'market_score': market_score,
            'location_score': location_score
        })

    except Exception as e:
        print(f"[ERROR] Error in /predict: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("="*50)
    print("Starting Startup Predictor")
    if model:
        print("[OK] Model loaded and cleaned")
    else:
        print("[WARN] Model not loaded. Using rule‑based fallback.")
    print("="*50)
    app.run(debug=True, port=5000)