import streamlit as st
import pandas as pd
# from src.compare_scores import plot_radar_chart
import seaborn as sns

st.title("Model Score Comparison")

eval_data = pd.read_csv("evaluation_results/model_score_comparison.csv")

st.write("Select models to include in the radar chart:")
models = eval_data['model'].unique()
selected_models = st.multiselect("Models", models)

st.write("Select concept to compare:")
concepts = eval_data['concept'].unique()
selected_concept = st.selectbox("Concept", concepts)

st.write("Select a text to focus on:")
texts = eval_data['label'].unique()
selected_text = st.selectbox("Text", texts)

st.write("Selected data")
filtered_data = eval_data[(eval_data['model'].isin(selected_models)) & (eval_data['concept'] == selected_concept)]
st.dataframe(filtered_data)




  

