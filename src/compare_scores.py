import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_all_model_scores(data):
    flat_data = []

    for item in data:
        label = item['label']
        for model_key, model_data in item['evaluations'].items():
            model = model_data['model_name']
            flat_data.append({
                'label': label,
                'model': model,
                'concept': 'TEXT',
                'dimension': '',
                'question': '',
                'score': model_data.get('overall_score')
            })

            for concept in model_data['concepts_scores']:
                concept_desc = concept['concept_description']
                flat_data.append({
                    'label': label,
                    'model': model,
                    'concept': concept_desc,
                    'dimension': '',
                    'question': '',
                    'score': concept.get('overall_score')
                })

                for dimension in concept['dimensions']:
                    dim_desc = dimension['dimension_description']
                    flat_data.append({
                        'label': label,
                        'model': model,
                        'concept': concept_desc,
                        'dimension': dim_desc,
                        'question': '',
                        'score': dimension.get('overall_score')
                    })

                    for question in dimension['questions']:
                        flat_data.append({
                            'label': label,
                            'model': model,
                            'concept': concept_desc,
                            'dimension': dim_desc,
                            'question': question['label'],
                            'score': question['score']
                        })

    return pd.DataFrame(flat_data)

def compare_datasets(df1, df2):
    df1['source'] = 'file1'
    df2['source'] = 'file2'
    combined = pd.concat([df1, df2])
    pivoted = combined.pivot_table(
        index=['label', 'model', 'concept', 'dimension', 'question'],
        columns='source',
        values='score'
    ).reset_index()

    pivoted['difference'] = pivoted['file2'] - pivoted['file1']
    return pivoted


def plot_radar_chart(df, concept, models_to_include, label_to_focus_on):
    # Filter the DataFrame
    df_filtered = df[(df['concept'] == concept) & (df['model'].isin(models_to_include)) & (df['label'] == label_to_focus_on)]

    # Get unique dimensions
    dimensions = df_filtered['dimension'].unique()
    dimensions = [dim for dim in dimensions if dim != '' and str(dim) != 'nan']  # Remove empty string and 'nan'
    num_dimensions = len(dimensions)

    # Create a dictionary to store scores for each model
    model_scores = {model: [np.nan] * num_dimensions for model in models_to_include}

    # Populate the model_scores dictionary
    for model in models_to_include:
        for i, dimension in enumerate(dimensions):
            # Get the score for the current model and dimension
            score = df_filtered[(df_filtered['model'] == model) & (df_filtered['dimension'] == dimension)]['score'].values
            # Assign the score if it exists, otherwise leave as NaN
            if len(score) > 0:
                model_scores[model][i] = score[0]

    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, num_dimensions, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Aesthetics: remove spines, set background color
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#F0F0F0')

    # Plot each model
    for model, scores in model_scores.items():
        # Replace NaN values with 0 for plotting
        scores = [0 if np.isnan(score) else score for score in scores]
        scores += scores[:1]  # Close the circle

        ax.plot(angles, scores, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, scores, alpha=0.25)

    # Set dimension labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)

    # Add a title and legend
    ax.set_title(f'Radar Chart for {concept} - {label_to_focus_on}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Show the plot
    #plt.show()
    return fig, ax
    

def normalize_and_export(df1, df2, save_path="model_score_comparison_flat.csv"):
    # Add source flag
    df1['source'] = 'file1'
    df2['source'] = 'file2'

    # Combine and rename
    combined = pd.concat([df1, df2])
    combined = combined.rename(columns={'score': 'score'})  # ensure standard naming

    # Drop unused columns if necessary
    combined = combined[['label', 'model', 'concept', 'dimension', 'question', 'score']]

    # Export
    combined.to_csv(save_path, index=False)
    print(f"✅ Normalized comparison saved to '{save_path}'")

    return combined


if __name__ == "__main__":
        # Load files
    data1 = load_json("evaluation_results/validation_data.json")
    data2 = load_json("evaluation_results/model_eval_data.json")

    # Extract scores
    df1 = extract_all_model_scores(data1)
    df2 = extract_all_model_scores(data2)

    # # Compare and export
    comparison_df = normalize_and_export(df1, df2)
    comparison_df = pd.read_csv("evaluation_results/model_score_comparison_flat.csv")

    # Plot radar chart for a specific concept and label
    concept = 'Taalniveau_B1'
    models = ["gpt-3.5-turbo-0125","gpt-4o", "gpt-4-turbo", "LL-01-pro"]
    label = 'B1 - Goed voorbeeld 2'
    plot_radar_chart(comparison_df, concept, models, label)
