import streamlit as st
import pandas as pd
from PIL import Image
# -------------------------------------------------------------------------------------
# Automation of workflows for proposal flow
st.header("3️⃣ Automation of workflows for proposal flow")

# ---------------------------------------------------
# Dummy Proposal Data
st.subheader("Dummy Proposal Data")
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
image = Image.open('work_flow.png')
st.image(image, caption='Work flow simulation', use_column_width=True)

# -------------------------------------------------------------------------------------
# Summary
st.success("A basic prototype to visualize data mapping, pricing model, and approval flow.")

