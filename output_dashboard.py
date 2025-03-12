import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.analysis import (
    load_evals_dict, 
    create_eval_df, 
    combine_eval_dfs, 
    calculate_average_question_score,
    calculate_average_dimension_score, 
    calculate_average_concept_score
)

# Set page configuration
st.set_page_config(
    page_title="Lazarsfeld LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Lazarsfeld LLM Evaluation Dashboard")
st.markdown("""
This dashboard visualizes the results of language model evaluations based on the Lazarsfeld framework.
Select files to compare and explore evaluation scores across concepts and dimensions.
""")

# Get available evaluation files
def get_available_files():
    files = glob.glob("evaluation_results/*.json")
    return [os.path.basename(f).replace(".json", "") for f in files]

# Sidebar for controls
st.sidebar.header("Controls")

# File selection
available_files = get_available_files()
selected_files = st.sidebar.multiselect(
    "Select evaluation files to compare:",
    options=available_files,
    default=available_files[:1] if available_files else []
)

if not selected_files:
    st.warning("Please select at least one evaluation file to continue.")
    st.stop()

# Load selected evaluations
with st.spinner("Loading evaluation data..."):
    evals = load_evals_dict(selected_files)
    
    # Create DataFrames from evaluations
    eval_dfs = {}
    for text, eval_result in zip(selected_files, evals):
        eval_dfs[text] = create_eval_df(eval_result)
    
    # Combine DataFrames for comparison
    if len(eval_dfs) > 0:
        combined_df = combine_eval_dfs(eval_dfs)

# Calculate average scores across models for each text
def calculate_cross_model_scores(df):
    # Group by text_key, concept, and dimension, then average across models
    return df.groupby(['text_key', 'concept', 'dimension']).agg({'score': 'mean'}).reset_index()

# Display tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", 
    "Dimension Comparison", 
    "Question Analysis", 
    "Raw Data"
])

# Tab 1: Overview
with tab1:
    st.header("Evaluation Overview")
    
    # Calculate average scores across models
    cross_model_scores = calculate_cross_model_scores(combined_df)
    
    # Overall scores by text (averaging across dimensions and models)
    overall_by_text = cross_model_scores.groupby('text_key').agg({'score': 'mean'}).reset_index()
    
    # Display as bar chart
    if not overall_by_text.empty:
        st.subheader("Overall Scores by Text")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='text_key', y='score', data=overall_by_text, ax=ax)
        plt.ylim(0, 1)
        plt.title("Overall Evaluation Scores by Text")
        plt.xlabel("Text")
        plt.ylabel("Average Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Dimension scores across texts
    st.subheader("Dimension Scores Across Texts")
    
    # Calculate average dimension scores
    avg_dim_scores = cross_model_scores.groupby(['dimension', 'text_key']).agg({'score': 'mean'}).reset_index()
    dim_pivot = avg_dim_scores.pivot(index='dimension', columns='text_key', values='score')
    
    # Display dimension scores as bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    dim_pivot.plot(kind='bar', ax=ax)
    plt.title("Average Dimension Scores Across Texts")
    plt.xlabel('Dimension')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)
    plt.legend(title='Text')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display as radar/spider chart if multiple dimensions
    if len(dim_pivot.index) >= 3:
        st.subheader("Dimension Profile (Radar Chart)")
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        
        # Number of dimensions
        categories = list(dim_pivot.index)
        N = len(categories)
        
        # Create angle for each dimension
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        
        # Add dimension labels
        plt.xticks(angles[:-1], categories)
        
        # Plot each text's dimension scores
        for text in dim_pivot.columns:
            values = dim_pivot[text].tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=text)
            ax.fill(angles, values, alpha=0.25)
        
        plt.title("Dimension Profile Across Texts")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        st.pyplot(fig)

# Tab 2: Dimension Comparison
with tab2:
    st.header("Dimension Score Comparison")
    
    # Model selection for dimension comparison
    model_options = list(set(combined_df["model"])) if not combined_df.empty else []
    selected_model = st.selectbox("Select Model for Comparison:", model_options)
    
    if selected_model:
        # Filter by selected model
        model_df = combined_df[combined_df["model"] == selected_model]
        
        # Calculate average dimension scores
        avg_dim_scores = calculate_average_dimension_score(model_df)
        
        # Pivot for comparison
        dim_pivot = avg_dim_scores.pivot(index='dimension', columns='text_key', values='score')
        
        # Plot
        st.subheader(f"Dimension Scores for {selected_model}")
        fig, ax = plt.subplots(figsize=(12, 8))
        dim_pivot.plot(kind='bar', ax=ax)
        plt.title(f"Average Dimension Scores - {selected_model}")
        plt.xlabel('Dimension')
        plt.ylabel('Average Score')
        plt.ylim(0, 1)
        plt.legend(title='Text')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display table of scores
        st.dataframe(dim_pivot, use_container_width=True)

# Tab 3: Question Analysis
with tab3:
    st.header("Question Level Analysis")
    
    # Controls for filtering questions
    col1, col2 = st.columns(2)
    
    with col1:
        dimension_options = sorted(list(set(combined_df["dimension"]))) if not combined_df.empty else []
        selected_dimension = st.selectbox("Filter by Dimension:", 
                                          options=["All"] + dimension_options)
    
    with col2:
        score_threshold = st.slider("Minimum Score Threshold:", 
                                   min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    
    # Filter data based on selections
    filtered_df = combined_df.copy()
    if selected_dimension != "All":
        filtered_df = filtered_df[filtered_df["dimension"] == selected_dimension]
    filtered_df = filtered_df[filtered_df["score"] >= score_threshold]
    
    # Calculate average question scores
    avg_question_scores = calculate_average_question_score(filtered_df)
    
    # Display question scores
    if not avg_question_scores.empty:
        st.subheader("Average Question Scores")
        
        # Create pivot table
        question_pivot = avg_question_scores.pivot(index='question', columns='text_key', values='score')
        
        # Sort by average score
        question_pivot['Avg'] = question_pivot.mean(axis=1)
        question_pivot = question_pivot.sort_values(by='Avg', ascending=False)
        
        # Remove the average column after sorting
        display_pivot = question_pivot.drop(columns=['Avg'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(display_pivot) * 0.3)))
        display_pivot.plot(kind='barh', ax=ax)
        plt.title("Average Question Scores")
        plt.xlabel('Score')
        plt.ylabel('Question')
        plt.xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display table of scores
        st.dataframe(display_pivot, use_container_width=True)

