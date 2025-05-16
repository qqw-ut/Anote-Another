import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
# -------------------------------------------------------------------------------------
# Inventory Pricing & Buyer Allocation(ML)
st.header("2️⃣ Automation of Inventory Pricing & Partner allocation(ML)")

# ----------------------------------
# Upload Excel for ML
dummy_buyers_file = st.file_uploader("Dummy Buyers Upload", type=["xlsx"])
if dummy_buyers_file:
    buyers_df=pd.read_excel(dummy_buyers_file)
    st.subheader("Dummy Buyers Upload Data (Raw)")
    st.dataframe(buyers_df)

# ----------------------------------
# ML Progress
X=buyers_df[['Buyer','Product Category','Historical Sales','Proposed Price','Target Margin %','Inventory Capacity']]
Y=buyers_df['Allocated Units']
categorical_cols=['Buyer','Product Category']
encoder=OneHotEncoder(sparse_output=False)
X_encoded=encoder.fit_transform(X[categorical_cols])
numerical_cols=['Historical Sales','Proposed Price','Target Margin %','Inventory Capacity']
X_all=np.hstack((X_encoded,X[numerical_cols]))
X_train,X_test,y_train,y_test=train_test_split(X_all,Y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

# ------------------------------------------------
# Input Buyer & Product Details
st.subheader("Input Buyer & Product Details For Allocation")
buyer = st.selectbox("Select Buyer", buyers_df['Buyer'].unique())
category = st.selectbox("Select Product Category", buyers_df['Product Category'].unique())
col1, col2 = st.columns(2)
with col1:
    historical_sales = st.number_input("Historical Sales", min_value=0, value=1000, step=100)
with col2:
    inventory_capacity = st.number_input("Inventory Capacity", min_value=0, value=3000, step=100)
col3, col4 = st.columns(2)
with col3:
    proposed_price = st.number_input("Proposed Price", min_value=0.0, value=50.0, step=1.0)
with col4:
    target_margin = st.number_input("Target Margin %", min_value=0.0, value=25.0, step=1.0)

# ------------------------------------------------
# Prediction
if st.button("Predict Allocated Units"):
    input_df = pd.DataFrame({
        'Buyer': [buyer],
        'Product Category': [category],
        'Historical Sales': [historical_sales],
        'Proposed Price': [proposed_price],
        'Target Margin %': [target_margin],
        'Inventory Capacity': [inventory_capacity]
    })

    X_input_cat = encoder.transform(input_df[categorical_cols])
    X_input_final = np.hstack((X_input_cat,input_df[numerical_cols]))
    prediction = model.predict(X_input_final)
    st.success(f"📦 Predicted Allocated Units: **{int(prediction[0])} units**")



# -------------------------------------------------------------------------------------
# Inventory Pricing & Buyer Allocation (Simple)
st.header("2️⃣ Automation of Inventory Pricing & Partner allocation (Simple)")

# ------------------------------------------------
# Dummy Data (matching their example)
st.subheader("Dummy data: Allocating 1000 units of Shoes to buyers")
data = [
    {'Product': 'Shoes', 'Buyer': 'Nike', 'Price': 20, 'Margin': 20, 'Historical Sales': 500},
    {'Product': 'Shoes', 'Buyer': 'Adidas', 'Price': 15, 'Margin': 25, 'Historical Sales': 300},
    {'Product': 'Shoes', 'Buyer': 'Reebok', 'Price': 25, 'Margin': 15, 'Historical Sales': 200},
]
df = pd.DataFrame(data)
st.dataframe(df)

# ------------------------------------------------
# Total Inventory (can be user input)
total_inventory = 1000 
df['Score'] = df['Price'] * (df['Margin'] / 100) * df['Historical Sales']
df['Allocation %'] = (df['Score'] / df['Score'].sum()) * 100
df['Allocated Units'] = (df['Allocation %'] / 100 * total_inventory).astype(int)
st.dataframe(df[['Product', 'Buyer', 'Price', 'Margin', 'Historical Sales', 'Score', 'Allocation %', 'Allocated Units']])


