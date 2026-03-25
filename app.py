from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Optional explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

# ==============================
# Paths / artifacts
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model_features.pkl')
TRAINING_X_NPY = os.path.join(BASE_DIR, 'train_X.npy')

# ==============================
# Globals
# ==============================
model = None
model_features = None
training_data_cache = None
lime_explainer_cache = None
shap_explainer_cache = None

# ==============================
# Helpers
# ==============================
CATEGORICAL_COLS = [
    'Employment_Status', 'Industry_Sector', 'Education_Level',
    'Marital_Status', 'Housing_Status', 'Credit_Mix',
    'Loan_Purpose', 'Collateral_Type', 'Seasonal_Spending_Pattern'
]

FLAG_COLS = ['Bankruptcy_Flag', 'Bankruptcy_Trigger_Flag']

NUMERIC_COLS = [
    'Age', 'Employment_Duration', 'Years_at_Residence',
    'Number_of_Dependents', 'Annual_Income', 'Total_Debt',
    'Debt_to_Income_Ratio', 'Loan_to_Income_Ratio', 'Credit_Score',
    'Credit_History_Length', 'Number_of_Existing_Loans',
    'Total_Credit_Limit', 'Credit_Utilization_Rate',
    'Savings_Account_Balance', 'Checking_Account_Balance',
    'Total_Assets', 'Net_Worth', 'Number_of_Late_Payments',
    'Worst_Delinquency_Status', 'Months_since_Last_Delinquency',
    'Number_of_Credit_Inquiries', 'Number_of_Open_Credit_Lines',
    'Number_of_Derogatory_Records', 'Loan_Amount_Requested',
    'Loan_Term_Months', 'Payment_to_Income_Ratio',
    'Collateral_Value', 'Transaction_Amount',
    'Transaction_Frequency', 'Days_since_Last_Transaction',
    'Avg_Probability_of_Default', 'Avg_Risk_Weighted_Assets',
    'DPD_Trigger_Count', 'Cash_Flow_Volatility'
]


def load_artifacts():
    global model, model_features
    try:
        model = joblib.load(MODEL_PATH)
        raw_features = joblib.load(FEATURES_PATH)

        if isinstance(raw_features, (pd.Index, np.ndarray)):
            model_features = list(map(str, list(raw_features)))
        elif isinstance(raw_features, list):
            model_features = list(map(str, raw_features))
        else:
            model_features = [str(x) for x in raw_features]

        print('✅ Model and features loaded successfully')
        print(f'📊 Features: {len(model_features)}')
        print(f'📊 Model type: {type(model).__name__}')
        return True
    except Exception as e:
        print(f'❌ Failed to load artifacts: {e}')
        model = None
        model_features = None
        return False


def load_training_data():
    global training_data_cache
    if training_data_cache is not None:
        return training_data_cache

    if os.path.exists(TRAINING_X_NPY):
        try:
            arr = np.load(TRAINING_X_NPY)
            training_data_cache = np.asarray(arr, dtype=float)
            print('✅ Loaded train_X.npy for LIME')
            return training_data_cache
        except Exception as e:
            print(f'⚠️ Could not load train_X.npy: {e}')

    return None


def normalize_bool(value):
    if isinstance(value, str):
        return 1 if value.strip().lower() in ('true', '1', 'yes', 'y') else 0
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except Exception:
        return 0


def preprocess_data(data):
    """Convert raw JSON input into a model-ready DataFrame."""
    df = pd.DataFrame([data])

    # Ensure expected categorical columns exist
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].astype(str).fillna('Unknown')

    # Ensure expected flag columns exist and are numeric 0/1
    for col in FLAG_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].apply(normalize_bool)

    # Ensure expected numeric columns exist; missing values become 0
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    if model_features is None:
        raise ValueError('model_features not loaded')

    # Add missing columns as zeros
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Drop extra columns not seen during training
    extra_cols = [c for c in df_encoded.columns if c not in model_features]
    if extra_cols:
        df_encoded = df_encoded.drop(columns=extra_cols)

    # Match training order exactly
    df_encoded = df_encoded[model_features]
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_encoded


def score_to_risk(score):
    """Frontend-aligned buckets."""
    if score >= 760:
        return 'Very Low Risk', 'green', 'approved', 'Approved - Excellent profile'
    elif score >= 660:
        return 'Low Risk', 'lightgreen', 'approved', 'Approved - Good profile'
    elif score >= 540:
        return 'Medium Risk', 'yellow', 'conditional', 'Conditional approval - review required'
    elif score >= 420:
        return 'High Risk', 'orange', 'rejected', 'Rejected - High risk'
    else:
        return 'Very High Risk', 'red', 'rejected', 'Rejected - Very high risk'


def predict_score(df_encoded):
    raw = model.predict(df_encoded)
    return float(np.clip(raw[0], 300, 900))


def get_shap_explanation(df_encoded):
    if not SHAP_AVAILABLE or model is None:
        return None

    try:
        if shap_explainer_cache is None:
            # TreeExplainer works well for XGBoost / tree-based models
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap_explainer_cache

        if shap_explainer_cache is None:
            globals()['shap_explainer_cache'] = explainer

        sample = df_encoded.values.astype(float)
        shap_values = explainer.shap_values(sample)
        expected_value = explainer.expected_value

        # Normalize shapes across SHAP versions/models
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        row_vals = shap_values[0]
        feature_values = df_encoded.iloc[0].values.astype(float)

        contributions = []
        for i, fname in enumerate(model_features):
            v = float(row_vals[i])
            contributions.append({
                'feature': fname,
                'shap_value': round(v, 6),
                'impact': 'positive' if v > 0 else 'negative' if v < 0 else 'neutral',
                'magnitude': abs(round(v, 6)),
                'actual_value': round(float(feature_values[i]), 6)
            })

        contributions.sort(key=lambda x: x['magnitude'], reverse=True)
        top_positive = [x for x in contributions if x['shap_value'] > 0][:5]
        top_negative = [x for x in contributions if x['shap_value'] < 0][:5]

        return {
            'method': 'SHAP',
            'base_value': float(np.squeeze(expected_value)) if np.size(expected_value) == 1 else float(np.ravel(expected_value)[0]),
            'all_features': contributions,
            'top_features': contributions[:10],
            'positive_features': top_positive,
            'negative_features': top_negative,
            'impact_distribution': {
                'risk_increasing_pct': round((len(top_positive) / max(len(top_positive) + len(top_negative), 1)) * 100, 2),
                'risk_reducing_pct': round((len(top_negative) / max(len(top_positive) + len(top_negative), 1)) * 100, 2)
            }
        }
    except Exception as e:
        return {
            'method': 'SHAP',
            'error': str(e),
            'all_features': [],
            'top_features': []
        }


def get_lime_explanation(df_encoded):
    if not LIME_AVAILABLE or model is None:
        return None

    try:
        data = load_training_data()
        if data is None:
            # fallback only if train_X.npy is missing
            base = np.tile(df_encoded.values[0], (500, 1))
            noise = np.random.normal(0.0, 1e-3, size=base.shape)
            data = base + noise

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(data, dtype=float),
            feature_names=model_features,
            mode='regression',
            discretize_continuous=True,
            verbose=False,
            random_state=42
        )

        exp = lime_explainer.explain_instance(
            data_row=df_encoded.values[0].astype(float),
            predict_fn=model.predict,
            num_features=10,
            num_samples=1000
        )

        lime_score = None
        try:
            if hasattr(exp, 'score'):
                lime_score = float(exp.score)
            elif hasattr(exp, 'local_pred'):
                lime_score = float(np.squeeze(exp.local_pred))
        except Exception:
            lime_score = None

        lime_contributions = []
        for feature_str, weight in exp.as_list():
            raw = feature_str.strip()
            feature_name = raw
            for sep in [' <= ', '<=', ' >= ', '>=', ' = ', '=', ' > ', '>', ' < ', '<']:
                if sep in raw:
                    feature_name = raw.split(sep)[0].strip()
                    break
            lime_contributions.append({
                'feature': feature_name,
                'weight': round(float(weight), 6),
                'impact': 'positive' if float(weight) > 0 else 'negative' if float(weight) < 0 else 'neutral',
                'magnitude': abs(round(float(weight), 6)),
                'description': feature_str
            })

        return {
            'method': 'LIME',
            'score': lime_score,
            'contributions': lime_contributions
        }
    except Exception as e:
        return {
            'method': 'LIME',
            'error': str(e),
            'contributions': []
        }


def generate_rejection_reasons(negative_features):
    reasons = []
    for feature in negative_features[:5]:
        name = feature['feature']
        clean = name.replace('_', ' ').title()
        if 'Credit_Score' in name:
            reasons.append({
                'factor': 'Credit Score',
                'issue': 'Low credit score is hurting approval chances',
                'improvement': 'Make timely payments and reduce utilization'
            })
        elif 'Debt_to_Income_Ratio' in name:
            reasons.append({
                'factor': 'Debt-to-Income Ratio',
                'issue': 'Debt is high relative to income',
                'improvement': 'Reduce debt or increase income'
            })
        elif 'Late_Payments' in name:
            reasons.append({
                'factor': 'Payment History',
                'issue': 'Late payment history is negatively impacting the score',
                'improvement': 'Keep all payments on time'
            })
        elif 'Credit_Utilization' in name:
            reasons.append({
                'factor': 'Credit Utilization',
                'issue': 'Credit utilization is too high',
                'improvement': 'Try to keep utilization below 30%'
            })
        elif 'Bankruptcy' in name:
            reasons.append({
                'factor': 'Bankruptcy History',
                'issue': 'Bankruptcy history is reducing creditworthiness',
                'improvement': 'Build positive credit history over time'
            })
        elif 'Annual_Income' in name or 'Income' in name:
            reasons.append({
                'factor': 'Income Level',
                'issue': 'Income may not support the requested loan',
                'improvement': 'Increase income or request a smaller loan'
            })
        else:
            reasons.append({
                'factor': clean,
                'issue': f'{clean} is negatively affecting the prediction',
                'improvement': f'Improve {clean.lower()} if possible'
            })
    return reasons


def generate_explanation_summary(status, rejection_reasons, approval_factors, score):
    if status == 'approved':
        summary = f'Your application has been approved with a credit risk score of {score:.2f}. '
        if approval_factors:
            top = ', '.join([x['factor'] for x in approval_factors[:3]])
            summary += f'Strong factors include {top}.'
        else:
            summary += 'Your profile looks favorable.'
        return summary

    if status == 'conditional':
        summary = f'Your application scored {score:.2f} and needs additional review. '
        if rejection_reasons:
            summary += f'The main concern is {rejection_reasons[0]["factor"]}. '
            summary += rejection_reasons[0]['improvement']
        return summary

    summary = f'Unfortunately, your application was declined with a score of {score:.2f}. '
    if rejection_reasons:
        top = ', '.join([x['factor'] for x in rejection_reasons[:3]])
        summary += f'Key issues: {top}. '
        summary += rejection_reasons[0]['improvement']
    return summary


# ==============================
# App setup
# ==============================
app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Credit Risk Prediction API with Explainability is running',
        'model_loaded': model is not None,
        'features_count': len(model_features) if model_features else 0,
        'explainability_methods': ['SHAP', 'LIME'],
        'shap_available': SHAP_AVAILABLE,
        'lime_available': LIME_AVAILABLE,
        'endpoints': {
            '/predict': 'Prediction only',
            '/explain': 'Prediction + SHAP + LIME'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': model_features is not None,
        'training_data_loaded': os.path.exists(TRAINING_X_NPY),
        'shap_available': SHAP_AVAILABLE,
        'lime_available': LIME_AVAILABLE
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if model is None or model_features is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        df_encoded = preprocess_data(data)
        score = predict_score(df_encoded)
        risk_level, risk_color, approval_status, approval_message = score_to_risk(score)

        return jsonify({
            'predicted_credit_risk_score': round(score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probability_of_default': round(max(0.02, min(0.50, (900 - score) / 1000)), 4),
            'approval_status': approval_status,
            'approval_message': approval_message,
            'total_features_used': len(df_encoded.columns),
            'status': 'success'
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'error_type': type(e).__name__, 'status': 'error'}), 500


@app.route('/explain', methods=['POST', 'OPTIONS'])
def explain():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if model is None or model_features is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        df_encoded = preprocess_data(data)
        score = predict_score(df_encoded)
        risk_level, risk_color, approval_status, approval_message = score_to_risk(score)

        shap_explanation = get_shap_explanation(df_encoded)
        lime_explanation = get_lime_explanation(df_encoded)

        positive_features = []
        negative_features = []
        if shap_explanation and 'all_features' in shap_explanation:
            positive_features = [x for x in shap_explanation['all_features'] if x['shap_value'] > 0][:3]
            negative_features = [x for x in shap_explanation['all_features'] if x['shap_value'] < 0][:5]

        approval_factors = [
            {
                'factor': x['feature'].replace('_', ' ').title(),
                'impact': 'positive',
                'contribution': abs(x['magnitude'])
            }
            for x in positive_features
        ]

        rejection_reasons = generate_rejection_reasons(negative_features) if approval_status != 'approved' else []
        summary = generate_explanation_summary(approval_status, rejection_reasons, approval_factors, score)

        if shap_explanation and 'all_features' in shap_explanation:
            inc = len([x for x in shap_explanation['all_features'] if x['shap_value'] > 0])
            dec = len([x for x in shap_explanation['all_features'] if x['shap_value'] < 0])
            total = max(inc + dec, 1)
            impact_distribution = {
                'risk_increasing_pct': round((inc / total) * 100, 2),
                'risk_reducing_pct': round((dec / total) * 100, 2)
            }
        else:
            impact_distribution = {'risk_increasing_pct': 0, 'risk_reducing_pct': 0}

        return jsonify({
            'predicted_credit_risk_score': round(score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probability_of_default': round(max(0.02, min(0.50, (900 - score) / 1000)), 4),
            'approval_status': approval_status,
            'approval_message': approval_message,
            'total_features_used': len(df_encoded.columns),
            'status': 'success',
            'feature_importance_explanation': shap_explanation,
            'lime_explanation': lime_explanation,
            'rejection_reasons': rejection_reasons,
            'approval_factors': approval_factors,
            'impact_distribution': impact_distribution,
            'explanation_summary': summary
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'error_type': type(e).__name__, 'status': 'error'}), 500


if __name__ == '__main__':
    load_artifacts()
    port = int(os.environ.get('PORT', 10000))
    print('\n' + '=' * 50)
    print('Starting Credit Risk API with Explainability')
    print('=' * 50)
    print(f'Port: {port}')
    print(f'Model: {MODEL_PATH}')
    print(f'Features: {FEATURES_PATH}')
    print(f'Training data: {TRAINING_X_NPY}')
    print('=' * 50 + '\n')
    app.run(host='0.0.0.0', port=port, debug=True)
