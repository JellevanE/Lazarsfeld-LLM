import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from concepts import TextEval


def load_evals_dict(texts:list) -> list[TextEval]:
    """Loads evaluation results from JSON files into a list of TextEval objects."""
    evals = []
    for text in texts:
        with open(f'evaluation_results/{text}.json', 'r') as f:
            evals.append(TextEval(json.load(f)))

    return evals


def create_eval_df(eval_results:TextEval) -> pd.DataFrame:
    """Creates a DataFrame from the evaluation results."""
    data = []
    
    for model, eval in eval_results['evaluations'].items():
        for concept in eval['concepts_scores']:
            for dimension in concept['dimensions']:
                for question in dimension['questions']:
                    data.append({
                        'model': model,
                        'concept': concept['concept_description'],
                        'dimension': dimension['dimension_description'],
                        #'question_label': question['label'],
                        'question': question['question'],
                        'score': question['score'],
                        'logprob': question['logprob'],
                    })

    return pd.DataFrame(data)


def combine_eval_dfs(eval_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combines multiple evaluation DataFrames into one for analysis"""
    combined_df = pd.concat(eval_dfs.values(), keys=eval_dfs.keys(), names=['text_key'])
    combined_df = combined_df.reset_index(level='text_key')
    return combined_df


def calculate_average_question_score(eval_df:pd.DataFrame) -> pd.DataFrame:
    """Calculates the average question score for each model and concept."""
    avg_scores = eval_df.groupby(['concept', 'dimension', 'question', 'text_key']).agg({'score': 'mean'}).reset_index()
    return avg_scores


def calculate_average_dimension_score(eval_df:pd.DataFrame) -> pd.DataFrame:
    """Calculates the average dimension score for each model and concept."""
    avg_scores = eval_df.groupby(['concept', 'dimension', 'text_key']).agg({'score': 'mean'}).reset_index()
    return avg_scores


def calculate_average_concept_score(eval_df:pd.DataFrame) -> pd.DataFrame:
    """Calculates the average concept score for each model."""
    avg_scores = eval_df.groupby(['concept', 'text_key']).agg({'score': 'mean'}).reset_index()
    return avg_scores


def plot_dimension_scores(eval_dfs: dict[str, pd.DataFrame]) -> None:
    """Plots the average dimension scores for different texts in the same figure."""
    plt.figure(figsize=(10, 6))

    combined_eval_dfs = combine_eval_dfs(eval_dfs)
    avg_scores = calculate_average_dimension_score(combined_eval_dfs)
    avg_scores = avg_scores.pivot(index=['dimension'], columns='text_key', values='score')
    avg_scores.plot(kind='bar', alpha=0.7, ax=plt.gca())

    plt.title('Average Dimension Scores')
    plt.xlabel('Dimension')
    plt.ylabel('Average Score')
    plt.legend(title='Text')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_question_scores(eval_dfs: dict[str, pd.DataFrame], dimension_label:str) -> None:
    """Plots the average question scores for different texts in the same figure."""
    plt.figure(figsize=(10, 6))

    combine_df = pd.concat(eval_dfs.values(), keys=eval_dfs.keys(), names=['text_key'])
    combine_df = combine_df.reset_index(level='text_key')
    if dimension_label:
        combine_df = combine_df[combine_df['dimension'] == dimension_label]
    avg_scores = calculate_average_question_score(combine_df)
    avg_scores = avg_scores.pivot(index=['question'], columns='text_key', values='score')

    # Create scatter plot
    #for column in avg_scores.columns:
        #plt.scatter(avg_scores.index, avg_scores[column], label=column, alpha=0.7)
        #plt.plot(avg_scores.index, avg_scores[column], alpha=0.7)

    avg_scores.plot(kind='bar', alpha=0.7, ax=plt.gca())
    plt.title('Average Question Scores')
    plt.xlabel('Question')
    plt.ylabel('Average Score')
    plt.legend(title='Text', loc='upper left')
    plt.xticks(rotation=45)
    plt.set_cmap("plasma")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    texts = ['epson_printer', 
             'bol_tafel', 
             'b1_voorbeeld', 
             'c1_voorbeeld']
    
    # Load the evaluation results
    evals = load_evals_dict(texts)
    
    # Create a DataFrame from the evaluation results
    evals_dfs = {}
    for text, eval in zip(texts, evals):
        evals_dfs[text] = create_eval_df(eval)

    plot_dimension_scores(evals_dfs)

    #plot_question_scores(evals_dfs, dimension_label="Samenhang")
    
    # Save the DataFrame to a CSV file
    #eval_df.to_csv('evaluation_results/eval_results.csv', index=False)








