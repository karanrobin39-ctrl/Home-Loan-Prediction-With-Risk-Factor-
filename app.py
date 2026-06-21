import streamlit as st
import pandas as pd
import pickle as pk
import webbrowser
import tempfile
import os
import math

# Load model and scaler safely
try:
    model = pk.load(open('model.pkl','rb'))
    scaler = pk.load(open('scaler.pkl','rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check paths.")
    st.stop()

# EMI Calculation Function
def calculate_emi(principal, annual_rate, years):
    """Calculate EMI using standard formula"""
    if years == 0:
        return 0
    
    monthly_rate = annual_rate / 12 / 100
    months = years * 12
    
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = (principal * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    
    return round(emi, 2)

st.header('Loan Prediction & Risk Management System')

# User Inputs
no_of_dep = st.slider('Number of dependents', 0, 5)
grad = st.selectbox('Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.number_input('Annual Income (₹)', min_value=0, max_value=10_000_000)
Loan_Amount = st.number_input('Loan Amount (₹)', min_value=0, max_value=10_000_0000)
Loan_Dur = st.slider('Loan Duration (in Months)', 0, 60)
Cibil = st.slider('CIBIL Score', 0, 1000)
Assets = st.number_input('Assets Value (₹)', min_value=0, max_value=10_000_000)

# EMI Calculation Input
interest_rate = st.slider('Expected Interest Rate (%)', min_value=5.0, max_value=15.0, value=10.0, step=0.5)

# Encoding categorical values
grad_s = 1 if grad == 'Graduated' else 0
emp_s = 1 if self_emp == 'Yes' else 0

if st.button("Predict Loan Approval"):
    # Calculate EMI
    loan_years = Loan_Dur / 12  # Convert months to years
    emi_amount = calculate_emi(Loan_Amount, interest_rate, loan_years)
    total_payment = emi_amount * Loan_Dur
    total_interest = total_payment - Loan_Amount
    emi_income_ratio = (emi_amount * 12 / Annual_Income) * 100 if Annual_Income > 0 else 0
    
    # Prepare data
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

    # Risk Score Calculation (Weighted)
    risk_score = 0
    # CIBIL weight
    if Cibil >= 750:
        risk_score += 0
    elif 650 <= Cibil < 750:
        risk_score += 20
    else:
        risk_score += 40

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

    # Create detailed HTML report with EMI
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
            .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .risk-low {{ color: green; font-weight: bold; }}
            .risk-medium {{ color: orange; font-weight: bold; }}
            .risk-high {{ color: red; font-weight: bold; }}
            .emi-section {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Loan Prediction Detailed Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="result {'approved' if predict[0] == 1 else 'rejected'}">
            <h2>Prediction: {'LOAN APPROVED ✅' if predict[0] == 1 else 'LOAN REJECTED ❌'}</h2>
            <h3>Approval Probability: {probability:.2f}%</h3>
        </div>
        
        <div class="emi-section">
            <h3>📊 EMI Calculation</h3>
            <p><strong>Monthly EMI:</strong> ₹{emi_amount:,.2f}</p>
            <p><strong>Interest Rate:</strong> {interest_rate}%</p>
            <p><strong>Total Interest Payable:</strong> ₹{total_interest:,.0f}</p>
            <p><strong>Total Payment (Principal + Interest):</strong> ₹{total_payment:,.0f}</p>
            <p><strong>EMI to Income Ratio:</strong> {emi_income_ratio:.1f}%</p>
            <p><strong>Loan Tenure:</strong> {Loan_Dur} months ({loan_years:.1f} years)</p>
        </div>
        
        <div class="metrics">
            <h3>Risk Assessment</h3>
            <p class="risk-{risk_level.split()[0].lower()}">Risk Level: {risk_level}</p>
            <p>Risk Score: {risk_score}/100</p>
            <p>{recommendation}</p>
        </div>
        
        <div class="metrics">
            <h3>Financial Metrics</h3>
            <p>Loan-to-Income Ratio: {loan_to_income:.2f}</p>
            <p>Asset Coverage Ratio: {asset_coverage:.2f}</p>
            <p>CIBIL Score: {Cibil}</p>
        </div>
        
        <div class="metrics">
            <h3>Applicant Details</h3>
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
    
    # Also show results in Streamlit with EMI
    st.markdown(f"### Prediction: {'Loan Approved' if predict[0] == 1 else 'Loan Rejected'}")
    st.markdown(f"**Approval Probability:** {probability:.2f}%")
    st.markdown(f"**Risk Level:** <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
    
    # Show EMI details in Streamlit
    st.markdown("---")
    st.markdown("### 📊 EMI Calculation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly EMI", f"₹{emi_amount:,.2f}")
        st.metric("Interest Rate", f"{interest_rate}%")
    with col2:
        st.metric("Total Interest", f"₹{total_interest:,.0f}")
        st.metric("EMI/Income Ratio", f"{emi_income_ratio:.1f}%")
    with col3:
        st.metric("Total Payment", f"₹{total_payment:,.0f}")
        st.metric("Loan Tenure", f"{Loan_Dur} months")
    
    st.success("📄 Detailed report opened in new tab!")
    
    # Show financial metrics
    st.write(f"Loan-to-Income Ratio: {loan_to_income:.2f}")
    st.write(f"Asset Coverage Ratio: {asset_coverage:.2f}")
    st.write(recommendation)