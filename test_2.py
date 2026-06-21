import streamlit as st
import pandas as pd
import pickle as pk
import webbrowser
import tempfile
import os

# Load model and scaler safely
try:
    model = pk.load(open('model.pkl','rb'))
    scaler = pk.load(open('scaler.pkl','rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check paths.")
    st.stop()

st.header('Loan Prediction & Risk Management System')

# User Inputs
no_of_dep = st.slider('Number of dependents', 0, 5)
grad = st.selectbox('Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.number_input('Annual Income (₹)', min_value=0, max_value=10_000_000)
Loan_Amount = st.number_input('Loan Amount (₹)', min_value=0, max_value=10_000_0000)
Loan_Dur = st.slider('Loan Duration (in Months)', 0, 60)

# NEW: Check if person has taken loans in the past
past_loans = st.selectbox('Has taken loans in the past?', ['No', 'Yes'])

# Conditionally show CIBIL score based on past loans
if past_loans == 'Yes':
    Cibil = st.slider('CIBIL Score', 300, 900, help="CIBIL score is required for applicants with past loan history")
    cibil_available = True
else:
    Cibil = 0  # Default value when no past loans
    cibil_available = False
    st.info("ℹ️ CIBIL score not required for first-time loan applicants")

Assets = st.number_input('Assets Value (₹)', min_value=0, max_value=10_000_000)

# Encoding categorical values
grad_s = 1 if grad == 'Graduated' else 0
emp_s = 1 if self_emp == 'Yes' else 0

if st.button("Predict Loan Approval"):
    
    # NEW BUSINESS RULE: For first-time applicants, check only income vs assets
    if past_loans == 'No':
        # Calculate income as percentage of assets
        income_to_assets_ratio = (Annual_Income / Assets) * 100 if Assets > 0 else 0
        
        st.markdown("### First-Time Applicant Special Assessment")
        st.write(f"**Income to Assets Ratio:** {income_to_assets_ratio:.2f}%")
        st.write(f"**Annual Income:** ₹{Annual_Income:,}")
        st.write(f"**Assets Value:** ₹{Assets:,}")
        
        # Apply the 50% rule
        if income_to_assets_ratio >= 50:
            manual_approval = True
            st.success("✅ **Rule Applied:** Income is 50% or more of assets value")
            st.success("✅ **Manual Decision:** LOAN APPROVED (First-time applicant rule)")
            
            # Set values for reporting
            predict = [1]
            probability = 85.0  # High probability for manual approval
            risk_score = 10
            risk_level = "Low Risk ✅"
            color = "green"
            recommendation = "✅ **Recommendation:** Loan approved under first-time applicant rule"
            rule_applied = "First-time applicant rule: Income ≥ 50% of assets"
        else:
            manual_approval = False
            st.error(f"❌ **Rule Applied:** Income ({income_to_assets_ratio:.2f}%) is less than 50% of assets value")
            st.error("❌ **Manual Decision:** LOAN REJECTED (First-time applicant rule)")
            
            # Set values for reporting
            predict = [0]
            probability = 15.0  # Low probability for manual rejection
            risk_score = 80
            risk_level = "High Risk ❌"
            color = "red"
            recommendation = "❌ **Recommendation:** Loan rejected - Income less than 50% of assets"
            rule_applied = "First-time applicant rule: Income < 50% of assets"
        
        cibil_status = "No CIBIL history (First-time applicant)"
        cibil_impact = "CIBIL score not considered (first-time applicant)"
        first_time_benefit = "First-time applicant special rules applied"
        
    else:
        # Original logic for applicants with past loans
        manual_approval = False
        rule_applied = "Standard assessment with CIBIL score"
        
        # Prepare data with ONLY the original 8 features that the model was trained on
        pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                 columns=['no_of_dependents','education','self_employed','income_annum',
                                          'loan_amount','loan_term','cibil_score','Assets'])
        
        # Scale
        pred_data_scaled = scaler.transform(pred_data)

        # Prediction & Probability
        predict = model.predict(pred_data_scaled)
        probability = model.predict_proba(pred_data_scaled)[0][1] * 100  # Probability of approval

        # Business Rule Calculations
        loan_to_income = Loan_Amount / Annual_Income if Annual_Income > 0 else 1
        asset_coverage = Assets / Loan_Amount if Loan_Amount > 0 else 0

        # Risk Score Calculation with CIBIL check
        risk_score = 0
        
        # CIBIL weight - for applicants with past loans
        if Cibil >= 750:
            risk_score += 0
        elif 650 <= Cibil < 750:
            risk_score += 20
        else:
            risk_score += 40
        cibil_status = f"CIBIL Score: {Cibil} (Historical data available)"
        cibil_impact = "CIBIL score considered in risk assessment"
        
        # Loan-to-Income weight
        if loan_to_income <= 0.4:
            risk_score += 0
        elif loan_to_income <= 0.6:
            risk_score += 20
        else:
            risk_score += 40

        # Asset coverage weight
        if asset_coverage >= 1:
            risk_score += 0
        elif asset_coverage >= 0.5:
            risk_score += 10
        else:
            risk_score += 20

        # ML model confidence adjustment
        if probability < 50:
            risk_score += 20

        first_time_benefit = "Applicant has previous loan history"

        # Classify Risk
        if risk_score <= 20:
            risk_level = "Low Risk ✅"
            color = "green"
            recommendation = "✅ **Recommendation:** Loan can be approved"
        elif risk_score <= 50:
            risk_level = "Medium Risk ⚠️"
            color = "orange"
            recommendation = "⚠️ **Recommendation:** Further verification required"
        else:
            risk_level = "High Risk ❌"
            color = "red"
            recommendation = "❌ **Recommendation:** Loan should be rejected"

    # Common calculations for reporting
    loan_to_income = Loan_Amount / Annual_Income if Annual_Income > 0 else 1
    asset_coverage = Assets / Loan_Amount if Loan_Amount > 0 else 0
    income_to_assets_ratio = (Annual_Income / Assets) * 100 if Assets > 0 else 0

    # Create detailed HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Loan Prediction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .result {{ padding: 20px; margin: 20px 0; border-radius: 10px; }}
            .approved {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
            .rejected {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
            .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .risk-low {{ color: green; font-weight: bold; }}
            .risk-medium {{ color: orange; font-weight: bold; }}
            .risk-high {{ color: red; font-weight: bold; }}
            .first-time {{ background-color: #e8f4fd; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; }}
            .rule-applied {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Loan Prediction Detailed Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        {f'<div class="first-time"><strong>🎉 First-time Loan Applicant</strong><br>Special assessment rules applied</div>' if past_loans == 'No' else ''}
        {f'<div class="rule-applied"><strong>📋 Business Rule Applied</strong><br>{rule_applied}</div>' if past_loans == 'No' else ''}
        
        <div class="result {'approved' if predict[0] == 1 else 'rejected'}">
            <h2>Prediction: {'LOAN APPROVED ✅' if predict[0] == 1 else 'LOAN REJECTED ❌'}</h2>
            <h3>Approval Probability: {probability:.2f}%</h3>
            {f'<p><strong>Decision Basis:</strong> {rule_applied}</p>' if past_loans == 'No' else ''}
        </div>
        
        <div class="metrics">
            <h3>Risk Assessment</h3>
            <p class="risk-{risk_level.split()[0].lower()}">Risk Level: {risk_level}</p>
            <p>Risk Score: {risk_score}/100</p>
            <p>{cibil_status}</p>
            <p>{recommendation}</p>
        </div>
        
        <div class="metrics">
            <h3>Financial Metrics</h3>
            <p>Loan-to-Income Ratio: {loan_to_income:.2f}</p>
            <p>Asset Coverage Ratio: {asset_coverage:.2f}</p>
            {'<p>Income to Assets Ratio: ' + f'{income_to_assets_ratio:.2f}%</p>' if past_loans == 'No' else ''}
        </div>
        
        <div class="metrics">
            <h3>Applicant Details</h3>
            <p><strong>Loan History:</strong> {past_loans}</p>
            <p>Annual Income: ₹{Annual_Income:,}</p>
            <p>Loan Amount: ₹{Loan_Amount:,}</p>
            <p>Loan Duration: {Loan_Dur} months</p>
            <p>Assets Value: ₹{Assets:,}</p>
            <p>Education: {grad}</p>
            <p>Self Employed: {self_emp}</p>
            <p>Dependents: {no_of_dep}</p>
        </div>
    </body>
    </html>
    """

    # Save HTML to temporary file and open in browser
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
        f.write(html_content)
        temp_file = f.name

    # Open in new tab
    webbrowser.open(f'file://{temp_file}', new=2)
    
    # Show results in Streamlit
    st.markdown(f"### Final Decision: {'Loan Approved' if predict[0] == 1 else 'Loan Rejected'}")
    
    if past_loans == 'No':
        st.markdown(f"**Decision Basis:** {rule_applied}")
        st.markdown(f"**Income to Assets Ratio:** {income_to_assets_ratio:.2f}%")
        if income_to_assets_ratio >= 50:
            st.success("✅ **Rule Met:** Annual income is 50% or more of total assets")
        else:
            st.error(f"❌ **Rule Not Met:** Annual income ({income_to_assets_ratio:.2f}%) is less than 50% of total assets")
    else:
        st.markdown(f"**Approval Probability:** {probability:.2f}%")
        st.markdown(f"**Risk Level:** <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
    
    st.success("📄 Detailed report opened in new tab!")
    
    # Show financial metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Annual Income", f"₹{Annual_Income:,}")
        st.metric("Assets Value", f"₹{Assets:,}")
    with col2:
        if past_loans == 'No':
            st.metric("Income to Assets Ratio", f"{income_to_assets_ratio:.2f}%")
            st.metric("Required Ratio", "≥50%")
        else:
            st.metric("CIBIL Score", Cibil)
            st.metric("Risk Score", f"{risk_score}/100")
    
    st.write(recommendation)