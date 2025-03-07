import numpy as np
import pandas as pd
import json
import matplotlib as plt
from concepts import TextEval

# load json data in TextEval format

texts = ['epson_printer', 
             'bol_tafel', 
             'b1_voorbeeld', 
             'c1_voorbeeld']
def load_evals_dict(texts:list) -> list[TextEval]:
    evals = []
    for text in texts:
        with open(f'evaluation_results/{text}.json', 'r') as f:
            evals.append(TextEval(json.load(f)))

    return evals
            
evals = load_evals_dict(texts)


# calculate average score for a text per concept across models
def calculate_avg_concept_score(evals:list[TextEval], concept_description:str) -> pd.DataFrame:
    concept_scores = []
    for eval in evals:
        for model_name, model_eval in eval['evaluations'].items():
            for concept_eval in model_eval['concepts_scores']:
                if concept_eval['concept_description'] == concept_description:
                    concept_scores.append({
                        'model_name': model_name,
                        'text': eval['input_text'],
                        'score': concept_eval['overall_score']
                    })
    df = pd.DataFrame(concept_scores)
    avg_scores = df.groupby('model_name')['score'].mean().reset_index()
    return avg_scores


    
#print(eval_data['input_text'])






