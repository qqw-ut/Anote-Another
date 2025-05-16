import streamlit as st
import pandas as pd

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