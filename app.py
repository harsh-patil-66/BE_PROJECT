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
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed")

# ==============================
# Load model and feature names
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model(2).pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")

try:
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print("✅ Model and features loaded successfully!")
    print(f"📊 Total features in model: {len(model_features)}")
    print(f"📊 Model type: {type(model).__name__}")
    
    # Check if model has feature_importances_
    if hasattr(model, 'feature_importances_'):
        print("✅ Model has feature_importances_ attribute")
    else:
        print("⚠️ Model does not have feature_importances_ attribute")
    
    explainer = None  # Will be set to None, feature importance will be used instead
    print("✅ Using Feature Importance for explanations (SHAP bypass)")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_features = None
    explainer = None

app = Flask(__name__)

# Enable CORS
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def preprocess_data(data):
    """Preprocess input data to match model requirements"""
    df = pd.DataFrame([data])

    # Categorical columns
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
                df[col] = 1 if val.upper() == "TRUE" else 0
            elif isinstance(val, bool):
                df[col] = 1 if val else 0
            else:
                df[col] = int(val) if val else 0

    # Convert numeric columns
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

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with training features
    missing_cols = [col for col in model_features if col not in df_encoded.columns]
    for col in missing_cols:
        df_encoded[col] = 0

    # Drop extra columns
    extra_cols = [col for col in df_encoded.columns if col not in model_features]
    if extra_cols:
        df_encoded = df_encoded.drop(columns=extra_cols)

    # Ensure correct column order
    df_encoded = df_encoded[model_features]
    
    # Ensure all values are numeric
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df_encoded

def get_feature_importance_explanation(df_encoded, prediction_value):
    """Generate feature importance based explanation (SHAP alternative)"""
    try:
        if hasattr(model, 'feature_importances_'):
            # Get feature importances from the model
            importances = model.feature_importances_
            
            # Get actual values from input
            input_values = df_encoded.values[0]
            
            # Calculate contribution (importance * value)
            contributions = []
            for i, feature_name in enumerate(model_features):
                importance = float(importances[i])
                value = float(input_values[i])
                
                # Contribution is importance weighted by the actual value
                contribution = importance * value
                
                contributions.append({
                    'feature': feature_name,
                    'shap_value': round(contribution, 4),
                    'impact': 'positive' if contribution > 0 else 'negative',
                    'magnitude': abs(round(contribution, 4)),
                    'feature_importance': round(importance, 4),
                    'actual_value': round(value, 4)
                })
            
            # Sort by magnitude
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
            # Fallback: use simple statistics
            print("⚠️ No feature_importances_ available, using value-based ranking")
            
            input_values = df_encoded.values[0]
            contributions = []
            
            for i, feature_name in enumerate(model_features):
                value = float(input_values[i])
                
                contributions.append({
                    'feature': feature_name,
                    'shap_value': round(value, 4),
                    'impact': 'positive' if value > 0 else 'negative',
                    'magnitude': abs(round(value, 4)),
                    'actual_value': round(value, 4)
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

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "✅ Credit Risk Prediction API with Explainability is running!",
        "model_loaded": model is not None,
        "features_count": len(model_features) if model_features else 0,
        "explainability_methods": ["Feature Importance", "LIME"],
        "shap_available": False,
        "version": "2.0.2",
        "endpoints": {
            "/predict": "Get prediction only",
            "/explain": "Get prediction with Feature Importance & LIME explanations"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_type": "feature_importance"
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Simple prediction without explainability"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None or model_features is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        print("\n" + "="*50)
        print("📥 Received prediction request (no explanations)")
        print("="*50)

        # Preprocess data
        df_encoded = preprocess_data(data)

        # Make prediction
        y_pred = model.predict(df_encoded)
        y_pred_value = float(np.clip(y_pred[0], 300, 900))

        print(f"🎯 Prediction: {y_pred_value}")

        # Determine risk level and approval status
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

        print("✅ Prediction complete (no explanations)")
        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR:\n{error_trace}\n")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        }), 500

@app.route('/explain', methods=['POST', 'OPTIONS'])
def explain():
    """Prediction WITH Feature Importance & LIME explanations"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None or model_features is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        print("\n" + "="*50)
        print("📥 Received EXPLAIN request (Feature Importance & LIME)")
        print("="*50)

        # Preprocess data
        df_encoded = preprocess_data(data)

        # Make prediction
        y_pred = model.predict(df_encoded)
        y_pred_value = float(np.clip(y_pred[0], 300, 900))

        print(f"🎯 Prediction: {y_pred_value}")

        # Determine risk level and approval status
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

        # Generate Feature Importance explanation
        print("🔄 Generating Feature Importance explanations...")
        shap_explanation = get_feature_importance_explanation(df_encoded, y_pred_value)
        
        if shap_explanation:
            print("✅ Feature Importance explanations generated")
        else:
            print("⚠️ Feature Importance generation failed")
        
        # Generate LIME explanations
        print("🔄 Generating LIME explanations...")
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.zeros((100, len(model_features))),
                feature_names=model_features,
                mode='regression',
                verbose=False
            )
            
            exp = lime_explainer.explain_instance(
                df_encoded.values[0],
                model.predict,
                num_features=10
            )
            
            lime_contributions = []
            for feature, weight in exp.as_list():
                feature_name = feature.split('<=')[0].split('>')[0].strip()
                lime_contributions.append({
                    'feature': feature_name,
                    'weight': round(weight, 4),
                    'impact': 'positive' if weight > 0 else 'negative',
                    'magnitude': abs(round(weight, 4)),
                    'description': feature
                })
            
            lime_explanation = {
                'contributions': lime_contributions,
                'score': float(exp.score),
                'explanation_type': 'LIME'
            }
            print("✅ LIME explanations generated")
        except Exception as lime_error:
            print(f"⚠️ LIME generation failed: {lime_error}")
            lime_explanation = {
                'contributions': [],
                'score': y_pred_value,
                'explanation_type': 'LIME',
                'error': str(lime_error)
            }

        # Generate rejection/approval reasons
        rejection_reasons = []
        approval_factors = []
        
        if shap_explanation:
            top_features = shap_explanation['top_features']
            
            if approval_status == "rejected" or approval_status == "conditional":
                negative_features = [f for f in top_features if f['impact'] == 'negative'][:5]
                rejection_reasons = generate_rejection_reasons(negative_features, y_pred_value)
            
            # Get positive factors
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
            
            # Explainability data
            "shap_explanation": shap_explanation,
            "lime_explanation": lime_explanation,
            "rejection_reasons": rejection_reasons,
            "approval_factors": approval_factors,
            
            # User-friendly summary
            "explanation_summary": generate_explanation_summary(
                approval_status, 
                rejection_reasons, 
                approval_factors, 
                y_pred_value
            )
        }

        print("✅ Explanation complete with Feature Importance & LIME")
        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR:\n{error_trace}\n")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        }), 500

def generate_rejection_reasons(negative_features, prediction_score):
    """Generate human-readable rejection reasons"""
    reasons = []
    
    for feature in negative_features:
        feature_name = feature['feature']
        
        # Map technical names to user-friendly explanations
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
                'issue': 'Your debt level is too high compared to your income',
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
    print(f"📍 Endpoints:")
    print(f"   - /predict (fast, no explanations)")
    print(f"   - /explain (with Feature Importance & LIME)")
    print(f"📊 Explanation Method: Feature Importance (SHAP alternative)")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)
