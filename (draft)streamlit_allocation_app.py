import streamlit as st
import pandas as pd

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

from PIL import Image
# -------------------------------------------------------------------------------------
# Page title
st.set_page_config(page_title="Another Inventory Automation", layout="wide")
st.title("📦 Inventory Proposal Automation - Prototype (Another)")

# ----------------------------------
# Upload Excel for mapping
st.header("1️⃣ Automation of inventory template mapping")
sample_sale_file = st.file_uploader("Sample Sale Upload", type=["xlsx"])
another_template_file = st.file_uploader("Another Template Upload", type=["xlsx"])

# ----------------------------------
# !Add a function for user to choose sheet
xls = pd.ExcelFile(sample_sale_file)
available_sheets = xls.sheet_names
input_sheet_name = st.selectbox('Select sheet in Sample Sale Upload', available_sheets)

# ----------------------------------
# Show uploaded files
if another_template_file and sample_sale_file:
    sample_df = pd.read_excel(sample_sale_file,sheet_name=input_sheet_name,header=1)
    sample_df = sample_df.dropna(axis=1,how='all') 
    # !Add a function to judge which column is pointless without any values
    another_df = pd.read_excel(another_template_file)
    st.subheader("Another Template Upload Data (Raw)")
    st.dataframe(another_df)
    st.subheader("Sample Sale Upload Data (Raw)")
    st.dataframe(sample_df)

    # ------------------------------------------
    # Dynamic User-defined Mapping(If the fields keep same in both file, just keep them)
    mapping_dict = {}
    for col in another_df.columns:
        c=[c for c in sample_df.columns if c.lower()==col.lower()]
        if c:
            selected_field=c[0]
        else:
            selected_field = st.selectbox(f"Map '{col}' to:", ["(None)"] + sample_df.columns.tolist(), key=col)
        if selected_field != "(None)":
            mapping_dict[selected_field] = col
    if mapping_dict:
        st.markdown("### 🗺️ Mapping Summary")
        st.json(mapping_dict)

        # ------------------------------------------
        # Apply Mapping & Generate Mapped Data
        renamed_sample_df=sample_df.rename(columns=mapping_dict) 
        # *Apply column renaming based on mapping_dict
        mapped_df =pd.DataFrame()
        for col in another_df.columns:
            if col in renamed_sample_df.columns:
                mapped_df[col]=renamed_sample_df[col]
            else:
                mapped_df[col] = 'TBD'
        st.subheader("📊 Mapped Data (Another Template Format)")
        st.dataframe(mapped_df)
        # ------------------------------------------
        # Download Mapped Result
        st.download_button(
            label="Download Mapped Another Template",
            data=mapped_df.to_csv(index=False).encode('utf-8'),
            file_name='mapped_another_template.csv',
            mime='text/csv'
        )

        st.subheader("📊 Mapped Data Editor (Another Template Format)")
        edit_mapped_df=st.data_editor(mapped_df)# !Add a editor function
        # ------------------------------------------
        # Download Edited Mapped Result
        st.download_button(
            label="Download Edited Mapped Another Template",
            data=edit_mapped_df.to_csv(index=False).encode('utf-8'),
            file_name='edit_mapped_another_template.csv',
            mime='text/csv'
        )
    else:
        st.warning("Please select field mappings to proceed.")
else:
    st.info("Please upload both a Source File and the Another Template File.")







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
st.subheader("Input Buyer & Product Details")
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





# -------------------------------------------------------------------------------------
# Automation of workflows for proposal flow
st.header("3️⃣ Automation of workflows for proposal flow")

# ---------------------------------------------------
# Dummy Proposal Data
data = [
    {'Buyer': 'Nike', 'Product': 'Shoes', 'Proposed Price': 20, 'Margin %': 20},
    {'Buyer': 'Adidas', 'Product': 'Shoes', 'Proposed Price': 15, 'Margin %': 25},
    {'Buyer': 'Reebok', 'Product': 'Shoes', 'Proposed Price': 25, 'Margin %': 15},
    {'Buyer': 'Puma', 'Product': 'Belts', 'Proposed Price': 10, 'Margin %': 30},
]
df = pd.DataFrame(data)
st.dataframe(df)

# ---------------------------------------------------
# User Approval Settings
st.subheader(" Set Approval Rules")
auto_approval_threshold = st.slider("Auto-approve proposals with Margin >= X%", min_value=0, max_value=50, value=25, step=1)
review_threshold = st.slider("Send to review if Margin >= Y% (but below auto-approve)", min_value=0, max_value=50, value=20, step=1)

st.markdown(f"🔹 Auto-approve if Margin >= {auto_approval_threshold}%")
st.markdown(f"🔹 Send to Review if Margin >= {review_threshold}% but < {auto_approval_threshold}%")
st.markdown(f"🔹 Reject if Margin < {review_threshold}%")

# ---------------------------------------------------
# Decision Logic
def decide_action(margin):
    if margin >= auto_approval_threshold:
        return '✅ Auto-Approved'
    elif margin >= review_threshold:
        return '📝 Needs Review'
    else:
        return '❌ Rejected'

df['Decision'] = df['Margin %'].apply(decide_action)

# ---------------------------------------------------
# Result Table
st.subheader("Proposal Approval Decisions")
st.dataframe(df[['Buyer', 'Product', 'Proposed Price', 'Margin %', 'Decision']])

# ---------------------------------------------------
#Simulate the Flow in Picture
#image = Image.open('example.jpg')
#st.image(image, caption='This is an example image', use_column_width=True)


# -------------------------------------------------------------------------------------
# Summary
st.success("A basic prototype to visualize data mapping, pricing model, and approval flow.")

