from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, but don't fail if it doesn't work
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed")

# ==============================
# Load model and feature names
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model(2).pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")
TRAINING_X_NPY = os.path.join(BASE_DIR, "train_X.npy")   # optional: saved training features

model = None
model_features = None
explainer = None

try:
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    # Ensure model_features is a list of strings
    if isinstance(model_features, (pd.Index, np.ndarray)):
        model_features = list(map(str, list(model_features)))
    elif isinstance(model_features, list):
        model_features = list(map(str, model_features))
    else:
        model_features = [str(x) for x in model_features]

    print("✅ Model and features loaded successfully!")
    print(f"📊 Total features in model: {len(model_features)}")
    print(f"📊 Model type: {type(model).__name__}")

    if hasattr(model, 'feature_importances_'):
        print("✅ Model has feature_importances_ attribute")
    else:
        print("⚠️ Model does not have feature_importances_ attribute")

except Exception as e:
    print(f"❌ Error loading model or features: {e}")
    model = None
    model_features = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["*"], "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})


def preprocess_data(data):
    """Preprocess a single input dict into a DataFrame matching model_features"""
    df = pd.DataFrame([data])

    # Categorical columns used during training (adjust to your real list)
    categorical_cols = [
        'Employment_Status', 'Industry_Sector', 'Education_Level',
        'Marital_Status', 'Housing_Status', 'Credit_Mix',
        'Loan_Purpose', 'Collateral_Type', 'Seasonal_Spending_Pattern'
    ]

    # Ensure all categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    # Convert boolean flags
    flag_cols = ["Bankruptcy_Flag", "Bankruptcy_Trigger_Flag"]
    for col in flag_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, str):
                df[col] = 1 if val.strip().lower() in ("true", "1", "yes") else 0
            elif isinstance(val, bool):
                df[col] = 1 if val else 0
            else:
                try:
                    df[col] = int(val)
                except Exception:
                    df[col] = 0

    # Numeric columns (adjust to your actual numeric features)
    numeric_cols = [
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

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            # ensure numeric column exists so one-hot alignment is simpler later
            df[col] = 0

    # One-hot encode categorical columns (drop_first to avoid multicollinearity)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with training features
    if model_features is None:
        raise ValueError("model_features is not loaded")

    # add missing columns (set to 0)
    missing_cols = [col for col in model_features if col not in df_encoded.columns]
    for col in missing_cols:
        df_encoded[col] = 0

    # drop extra columns not in model_features (safety)
    extra_cols = [col for col in df_encoded.columns if col not in model_features]
    if extra_cols:
        df_encoded = df_encoded.drop(columns=extra_cols)

    # ensure correct order and numeric types
    df_encoded = df_encoded[model_features]
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df_encoded


def get_feature_importance_explanation(df_encoded, prediction_value):
    """Generate feature importance based explanation (SHAP alternative)"""
    try:
        if model is None or model_features is None:
            return None

        input_values = df_encoded.values[0]
        contributions = []

        if hasattr(model, 'feature_importances_'):
            importances = np.array(model.feature_importances_, dtype=float)
            if importances.shape[0] != len(model_features):
                # guard mismatch
                print("⚠️ feature_importances_ length mismatch; using absolute importance ranking fallback")
                importances = np.abs(importances[:len(model_features)])
                # if still mismatch, fallback to equal importance
                if importances.shape[0] != len(model_features):
                    importances = np.ones(len(model_features)) / len(model_features)

            # contribution = importance * value (simple heuristic)
            for i, feature_name in enumerate(model_features):
                importance = float(importances[i])
                value = float(input_values[i])
                contribution = importance * value
                contributions.append({
                    'feature': feature_name,
                    'shap_value': round(contribution, 6),
                    'impact': 'positive' if contribution > 0 else 'negative' if contribution < 0 else 'neutral',
                    'magnitude': abs(round(contribution, 6)),
                    'feature_importance': round(importance, 6),
                    'actual_value': round(value, 6)
                })

            contributions.sort(key=lambda x: x['magnitude'], reverse=True)
            top_features = contributions[:10]

            explanation = {
                'all_features': contributions,
                'top_features': top_features,
                'base_value': float(prediction_value),
                'explanation_type': 'Feature Importance (Tree-based)',
                'method': 'model.feature_importances_'
            }
            return explanation

        else:
            # fallback: rank by absolute value of input features
            for i, feature_name in enumerate(model_features):
                value = float(input_values[i])
                contributions.append({
                    'feature': feature_name,
                    'shap_value': round(value, 6),
                    'impact': 'positive' if value > 0 else 'negative' if value < 0 else 'neutral',
                    'magnitude': abs(round(value, 6)),
                    'actual_value': round(value, 6)
                })
            contributions.sort(key=lambda x: x['magnitude'], reverse=True)
            top_features = contributions[:10]

            explanation = {
                'all_features': contributions,
                'top_features': top_features,
                'base_value': float(prediction_value),
                'explanation_type': 'Value-based Importance',
                'method': 'input_values'
            }
            return explanation

    except Exception as e:
        print(f"⚠️ Feature importance calculation failed: {e}")
        return None


def build_lime_training_data(single_sample_array, n_samples=500, noise_scale=1e-3):
    """
    Build a pseudo training set for LIME if no real training X is available.
    We tile the single sample and add tiny gaussian noise so LIME can fit a local surrogate.
    """
    n_features = single_sample_array.shape[0]
    base = np.tile(single_sample_array, (n_samples, 1))
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=base.shape)
    training_data = base + noise
    return training_data


@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "✅ Credit Risk Prediction API with Explainability is running!",
        "model_loaded": model is not None,
        "features_count": len(model_features) if model_features else 0,
        "explainability_methods": ["Feature Importance", "LIME"],
        "shap_available": SHAP_AVAILABLE,
        "version": "2.0.2",
        "endpoints": {
            "/predict": "Get prediction only",
            "/explain": "Get prediction with Feature Importance & LIME"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_type": "feature_importance + lime (if available)"
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if model is None or model_features is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df_encoded = preprocess_data(data)
        y_pred = model.predict(df_encoded)
        # keep your clipping if needed for score range
        y_pred_value = float(np.clip(y_pred[0], 300, 900))

        # risk buckets (same as your original logic)
        if y_pred_value >= 750:
            risk_level = "🟢 Excellent Credit"
            probability = 0.02
            approval_status = "approved"
            approval_message = "Loan Approved with Excellent Terms"
        elif y_pred_value >= 700:
            risk_level = "🟢 Good Credit"
            probability = 0.05
            approval_status = "approved"
            approval_message = "Loan Approved with Good Terms"
        elif y_pred_value >= 650:
            risk_level = "🟡 Fair Credit"
            probability = 0.15
            approval_status = "conditional"
            approval_message = "Conditional Approval - Additional Review Required"
        elif y_pred_value >= 600:
            risk_level = "🟠 Poor Credit"
            probability = 0.30
            approval_status = "rejected"
            approval_message = "Loan Application Rejected - High Risk"
        else:
            risk_level = "🔴 Very Poor Credit"
            probability = 0.50
            approval_status = "rejected"
            approval_message = "Loan Application Rejected - Very High Risk"

        response = {
            "predicted_credit_risk_score": round(y_pred_value, 2),
            "risk_level": risk_level,
            "probability_of_default": probability,
            "approval_status": approval_status,
            "approval_message": approval_message,
            "total_features_used": len(df_encoded.columns),
            "status": "success"
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR:\n{error_trace}\n")
        return jsonify({"error": str(e), "error_type": type(e).__name__, "status": "error"}), 500


@app.route('/explain', methods=['POST', 'OPTIONS'])
def explain():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if model is None or model_features is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df_encoded = preprocess_data(data)
        y_pred = model.predict(df_encoded)
        y_pred_value = float(np.clip(y_pred[0], 300, 900))

        # risk buckets (same as predict)
        if y_pred_value >= 750:
            risk_level = "🟢 Excellent Credit"
            probability = 0.02
            approval_status = "approved"
            approval_message = "Loan Approved with Excellent Terms"
        elif y_pred_value >= 700:
            risk_level = "🟢 Good Credit"
            probability = 0.05
            approval_status = "approved"
            approval_message = "Loan Approved with Good Terms"
        elif y_pred_value >= 650:
            risk_level = "🟡 Fair Credit"
            probability = 0.15
            approval_status = "conditional"
            approval_message = "Conditional Approval - Additional Review Required"
        elif y_pred_value >= 600:
            risk_level = "🟠 Poor Credit"
            probability = 0.30
            approval_status = "rejected"
            approval_message = "Loan Application Rejected - High Risk"
        else:
            risk_level = "🔴 Very Poor Credit"
            probability = 0.50
            approval_status = "rejected"
            approval_message = "Loan Application Rejected - Very High Risk"

        # Feature importance explanation
        print("🔄 Generating Feature Importance explanations...")
        fi_explanation = get_feature_importance_explanation(df_encoded, y_pred_value)
        if fi_explanation:
            print("✅ Feature Importance explanations generated")
        else:
            print("⚠️ Feature Importance generation failed")

        # LIME explanation
        print("🔄 Generating LIME explanations...")
        lime_explanation = None
        try:
            # attempt to load real training data for LIME if present
            try:
                if os.path.exists(TRAINING_X_NPY):
                    training_data = np.load(TRAINING_X_NPY)
                    print("ℹ️ Loaded training data for LIME from train_X.npy")
                else:
                    # synthesize a local training set from the input
                    training_data = build_lime_training_data(df_encoded.values[0], n_samples=500, noise_scale=1e-3)
                    print("ℹ️ No saved training set found — using synthesized local training data for LIME")
            except Exception as e_load:
                print(f"⚠️ Could not load saved training data: {e_load}. Using synthesized data.")
                training_data = build_lime_training_data(df_encoded.values[0], n_samples=500, noise_scale=1e-3)

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=model_features,
                mode='regression',
                discretize_continuous=True,
                verbose=False,
                random_state=42
            )

            # explain_instance: model.predict must accept 2D array
            exp = lime_explainer.explain_instance(
                df_encoded.values[0],
                model.predict,
                num_features=10,
                num_samples=1000  # more samples -> better surrogate fit
            )

            # Try to pick a sensible value for LIME's "score"
            lime_score = None
            if hasattr(exp, 'score'):
                try:
                    lime_score = float(exp.score)
                except Exception:
                    lime_score = None
            elif hasattr(exp, 'local_pred'):
                try:
                    lime_score = float(np.squeeze(exp.local_pred))
                except Exception:
                    lime_score = None
            if lime_score is None:
                lime_score = float(y_pred_value)

            # parse contributions
            lime_contributions = []
            for feature_str, weight in exp.as_list():
                # feature_str examples: "Credit_Score > 650.0", "Employment_Status=Employed"
                # extract feature name robustly (stop at first space or comparator)
                # remove trailing spaces
                raw = feature_str.strip()
                # split on common comparators/space to get feature token
                for sep in [' <= ', '<=', ' >= ', '>=', ' = ', '=', ' > ', '>', ' < ', '<', ' ']:
                    if sep in raw:
                        feature_name = raw.split(sep)[0].strip()
                        break
                else:
                    feature_name = raw  # fallback

                lime_contributions.append({
                    'feature': feature_name,
                    'weight': round(float(weight), 6),
                    'impact': 'positive' if float(weight) > 0 else 'negative' if float(weight) < 0 else 'neutral',
                    'magnitude': abs(round(float(weight), 6)),
                    'description': feature_str
                })

            lime_explanation = {
                'contributions': lime_contributions,
                'score': lime_score,
                'explanation_type': 'LIME'
            }
            print("✅ LIME explanations generated")
        except Exception as lime_error:
            print(f"⚠️ LIME generation failed: {lime_error}")
            lime_explanation = {
                'contributions': [],
                'score': float(y_pred_value),
                'explanation_type': 'LIME',
                'error': str(lime_error)
            }

        # Build human readable reasons (reuse your helpers)
        rejection_reasons = []
        approval_factors = []
        if fi_explanation:
            top_features = fi_explanation.get('top_features', [])
            if approval_status in ("rejected", "conditional"):
                negative_features = [f for f in top_features if f['impact'] == 'negative'][:5]
                rejection_reasons = generate_rejection_reasons(negative_features, y_pred_value)
            # positive factors
            positive_features = [f for f in top_features if f['impact'] == 'positive'][:3]
            for feature in positive_features:
                clean_name = feature['feature'].replace('_', ' ').title()
                approval_factors.append({
                    'factor': clean_name,
                    'impact': 'positive',
                    'contribution': feature['magnitude']
                })

        response = {
            "predicted_credit_risk_score": round(y_pred_value, 2),
            "risk_level": risk_level,
            "probability_of_default": probability,
            "approval_status": approval_status,
            "approval_message": approval_message,
            "total_features_used": len(df_encoded.columns),
            "status": "success",

            # Explainability
            "feature_importance_explanation": fi_explanation,
            "lime_explanation": lime_explanation,
            "rejection_reasons": rejection_reasons,
            "approval_factors": approval_factors,
            "explanation_summary": generate_explanation_summary(approval_status, rejection_reasons, approval_factors, y_pred_value)
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR:\n{error_trace}\n")
        return jsonify({"error": str(e), "error_type": type(e).__name__, "status": "error"}), 500


# (Keep your existing helper functions unchanged or copy them from your original code)
def generate_rejection_reasons(negative_features, prediction_score):
    """Generate human-readable rejection reasons"""
    reasons = []
    for feature in negative_features:
        feature_name = feature['feature']
        if 'Credit_Score' in feature_name:
            reasons.append({
                'factor': 'Credit Score',
                'issue': 'Your credit score is negatively impacting your application',
                'improvement': 'Work on improving your credit score by making timely payments and reducing credit utilization',
                'impact_score': feature['magnitude']
            })
        elif 'Debt_to_Income_Ratio' in feature_name:
            reasons.append({
                'factor': 'Debt-to-Income Ratio',
                'issue': 'Your debt level is high compared to your income',
                'improvement': 'Consider reducing existing debt or increasing income before reapplying',
                'impact_score': feature['magnitude']
            })
        elif 'Late_Payments' in feature_name:
            reasons.append({
                'factor': 'Payment History',
                'issue': 'History of late payments is affecting your application',
                'improvement': 'Make all future payments on time to rebuild payment history',
                'impact_score': feature['magnitude']
            })
        elif 'Credit_Utilization' in feature_name:
            reasons.append({
                'factor': 'Credit Utilization',
                'issue': 'You are using too much of your available credit',
                'improvement': 'Try to keep credit utilization below 30%',
                'impact_score': feature['magnitude']
            })
        elif 'Bankruptcy' in feature_name:
            reasons.append({
                'factor': 'Bankruptcy History',
                'issue': 'Previous bankruptcy is impacting your creditworthiness',
                'improvement': 'Continue building positive credit history over time',
                'impact_score': feature['magnitude']
            })
        elif 'Annual_Income' in feature_name or 'Income' in feature_name:
            reasons.append({
                'factor': 'Income Level',
                'issue': 'Your income level may not support the requested loan amount',
                'improvement': 'Consider requesting a smaller loan amount or demonstrating additional income sources',
                'impact_score': feature['magnitude']
            })
        elif 'Employment' in feature_name:
            reasons.append({
                'factor': 'Employment History',
                'issue': 'Limited employment history is affecting your application',
                'improvement': 'Build more stable employment history before reapplying',
                'impact_score': feature['magnitude']
            })
        elif 'Delinquency' in feature_name:
            reasons.append({
                'factor': 'Delinquency History',
                'issue': 'Past delinquencies are negatively impacting your application',
                'improvement': 'Focus on maintaining clean credit for at least 6-12 months',
                'impact_score': feature['magnitude']
            })
        else:
            clean_name = feature_name.replace('_', ' ').title()
            reasons.append({
                'factor': clean_name,
                'issue': f'{clean_name} is negatively impacting your application',
                'improvement': f'Improve your {clean_name.lower()} to increase approval chances',
                'impact_score': feature['magnitude']
            })
    return reasons


def generate_explanation_summary(approval_status, rejection_reasons, approval_factors, score):
    """Generate a plain English summary"""
    if approval_status == "approved":
        summary = f"Your application has been approved with a credit risk score of {score:.2f}. "
        summary += "Your strong financial profile, particularly "
        if approval_factors:
            factors = ", ".join([f['factor'] for f in approval_factors[:2]])
            summary += f"your {factors}, "
        summary += "demonstrates low default risk."
    elif approval_status == "conditional":
        summary = f"Your application received a score of {score:.2f}, which requires additional review. "
        if rejection_reasons:
            main_issue = rejection_reasons[0]['factor']
            summary += f"The main concern is your {main_issue}. "
            summary += rejection_reasons[0]['improvement']
    else:  # rejected
        summary = f"Unfortunately, your application was declined with a credit risk score of {score:.2f}. "
        if rejection_reasons:
            summary += "The primary reasons include: "
            reasons_list = [r['factor'] for r in rejection_reasons[:3]]
            summary += ", ".join(reasons_list) + ". "
            summary += f"To improve: {rejection_reasons[0]['improvement']}"
    return summary


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print("\n" + "="*50)
    print("🚀 Starting Credit Risk API with Explainability")
    print("="*50)
    print(f"📍 Server: http://localhost:{port}")
    print(f"📍 Endpoints: /predict, /explain")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=port, debug=True)
