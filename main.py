import streamlit as st
import pandas as pd
import processing

Data_Base_Names = {
    "names": [
    'Bluewave Solutions Ltd',
'SwiftPeak Technologies Inc',
'Evergreen Ventures LLC',
'CrystalEdge Innovations',
'Lunar Horizon Enterprises',
'Silverline Analytics',
'Brightpath Consulting Group',
'Vortex Dynamics Inc',
'Aurora Nexus Holdings',
'Ironclad Security Systems',
'Crestline Developers Limited',
'Nexon BioTech LLC',
'VelvetStream Apparel Co',
'UrbanOrbit Interiors',
'Pioneer Energy Solutions',
'StellarCore Technologies',
'MapleCrest Foods LLC',
'Quantum Ridge Consulting',
'Oceanic Breeze Travel Co',
'Timberland Industrial Supplies',
'Zenith Health Partners',
'AmberField Construction Group',
'NextEra EcoWorks',
'Fusion Point Media',
'Skyline Horizons Inc',
'Four Seasons'
],
}
Prohibited_Words = {
    "prohibited_words": ["bank", "police", "government","unauthorized"],
}

df1 = pd.DataFrame(Data_Base_Names)
df2 = pd.DataFrame(Prohibited_Words)
prohibited_words = df2['prohibited_words'].to_list()
# Streamlit application
st.title("SECP")

# Bold label for the input field
st.header("Enter Company Name:")
user_input = st.text_input("", "")  # Empty label to avoid duplicate text
# Output field
st.header("Matching Result:")
output_text = st.text_area("",value=processing.get_similarities(user_input,df1['names'].to_list()), height=160)

# Display DataFrames side by side
col1, col2 = st.columns(2)

with col1:
    st.write("Data Base Names")
    st.dataframe(df1)

with col2:
    st.write("Prohibited Words")
    st.dataframe(df2)
