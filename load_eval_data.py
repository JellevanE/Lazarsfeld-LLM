import os
import pickle
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config.config import CREDS, TOKEN_FILE, SCOPES, SHEET_ID, SHEET_NAMES, DEFAULT_WEIGHT
from config.evaluation_config import BOORMACHINE_ADVICE_TEXT

from src.concepts import QuestionEval, DimensionEval, ConceptEval, TextEval, ModelEval, Concept, Dimension, Question, ValidationScores
from src.update_concepts import process_concept_csv
from src.utils import fancy_print_output


def authenticate():
    """Authenticate and return Google API credentials."""
    creds = None

    # Load previously stored token if available
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, go through OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
            creds = flow.run_local_server(port=0)  # Opens a browser for login

        # Save credentials for future use
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    return creds


def load_eval_scores_from_sheet(sheet_id: str, sheet_name: str) -> dict[str:int]:
    """Loads evaluation scores from a Google Sheet."""
    creds = authenticate()
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    # Strip whitespace and handle potential encoding issues
    sheet_name = sheet_name.strip()
    range_name =  f"{sheet_name}!A1:I99"  # Adjust range as needed

    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()

    values = result.get("values", [])
    if not values:
        raise ValueError("No data found in the sheet.")
    
    # Add data from column Label and Score into dict with label as key and score as value
    scores = {}
    for row in values[1:]:
        if len(row) >= 3:  # Check if the row has at least 3 elements
            label = row[0]
            score = row[2]
            if label and score:
                scores[label] = int(score)
        else:
            print(f"Skipping row due to insufficient elements: {row}")  # Optional: Log the skipped row

    return scores


# Function to evaluate a single question using LLM
def evaluate_question(
        question_obj: Question, 
        eval_scores:dict[str:int]
        ) -> QuestionEval:

    # Extract question details
    question = question_obj["question"]
    positive_contribution = question_obj["positive_contribution"]
    if question_obj["positive_contribution"] == True:
        token = "True"
        probability = (eval_scores.get(question_obj["label"], None)-1)/4
    elif question_obj["positive_contribution"] == False:
        token = "False"
        probability = 1 - (eval_scores.get(question_obj["label"], None)-1)/4
    else:
        token = None
    
    return {
        "label": question_obj["label"],
        "question": question,
        "answer": token,
        "score": probability,
        "logprob": np.log(probability + 1e-9),
        "positive_contribution": positive_contribution
    }


# Function to evaluate a single dimension using LLM
def evaluate_dimension(
        dimension: Dimension, 
        eval_scores:dict[str:int]
        ) -> DimensionEval:
    
    questions = dimension["questions"]
    question_scores = []

    for question in questions:
        question_eval = evaluate_question(question, eval_scores)
        question_scores.append(question_eval)

    # Calculate overall score for the dimension, excluding None values
    scores = [q["score"] for q in question_scores if q["score"] is not None]
    overall_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    return {
        "dimension_description": dimension["dimension_description"],
        "questions": question_scores,
        "overall_score": overall_score,
        "weight": dimension.get("weight", DEFAULT_WEIGHT)
    }


# Function to evaluate a single concept using LLM
def evaluate_concept(
        concept: Concept, 
        eval_scores:dict[str:int]
        ) -> ConceptEval:
    
    dimensions = concept["dimensions"]
    dimension_scores = []

    for dimension in dimensions:
        dimension_eval = evaluate_dimension(dimension, eval_scores)
        dimension_scores.append(dimension_eval)

    # Calculate overall score for the concept, excluding None values
    scores = [d["overall_score"] for d in dimension_scores if d["overall_score"] is not None]
    overall_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    return {
        "concept_description": concept["concept_description"],
        "dimensions": dimension_scores,
        "overall_score": overall_score,
        "weight": concept.get("weight", DEFAULT_WEIGHT)
    }


def model_eval(
        model: str, 
        concepts: list[Concept], 
        label:str,
        eval_scores:dict[str:int]
        ) -> ModelEval:
    
    # Initialize OpenAI client
    print(f"\nEvaluating text: {label} \nUsing model: {model}\n\n")
    
    concept_scores = []
    for concept in concepts:
        concept_eval = evaluate_concept(concept, eval_scores)
        concept_scores.append(concept_eval)
    
    # Calculate overall score for the model, excluding None values
    scores = [c["overall_score"] for c in concept_scores if c["overall_score"] is not None]
    overall_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    return {
        "model_name": model,
        "concepts_scores": concept_scores,
        "overall_score": overall_score,
        "weight": DEFAULT_WEIGHT
    }


def text_eval( 
        models:list[str], 
        text:str,
        label:str,
        concepts:list[Concept],
        eval_scores:dict[str:int]
        ) -> TextEval:
    
    evaluations = {}
    # Evaluate the text using each model using threads
    for model in models:
        model_eval_result = model_eval(model, concepts, label, eval_scores)
        evaluations[model] = model_eval_result

    # Calculate aggregated score for the text, excluding None values
    scores = [m["overall_score"] for m in evaluations.values() if m and m["overall_score"] is not None]
    aggregated_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    metadata = {
        "models_used": models,
        "evaluation_parameters": {
            "system_prompt": str("Geef eens score tussen 1 en 5"),
            "base_prompt": str("Laurence & Lieze")
        }
    }

    timestamp = datetime.now().strftime("%Y-%m-%d")


    return {
        "label": label,
        "input_text": text,
        "concepts": concepts,
        "evaluations": evaluations,
        "aggregated_score": aggregated_score,
        "metadata": metadata,
        "timestamp": timestamp
    }


def main(texts:dict, models:list, concepts:list[Concept], output_dir, eval_scores:dict) -> None:
    """Runs evaluation pipeline and saves results to JSON file."""
    
    results = []
    # Run evaluation
    for label, text in texts.items():
        text_eval_result = text_eval(models, text, label, concepts, eval_scores)
        results.append(text_eval_result)
        print(f"Evaluation for {label} completed.\n")

    # Save results to JSON file
    with open(f"{output_dir}{label}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print results to console
    fancy_print_output(text_eval_result)



if __name__ == "__main__":
    # Load eval concepts from CSV
    csv_path = Path('eval_concepts/LLM_eval_concepten - Taalniveau B1.csv')
    process_concept_csv(csv_path, output_filepath=Path('eval_concepts/taalniveau_b1_concept.json'), concept_name="Taalniveau_B1")

    with open ("eval_concepts/taalniveau_b1_concept.json", "r") as f:
        concepts = json.load(f)

    output_dir = "evaluation_results/"

    # Load evaluation scores from Google Sheets
    scores = {}
    for name in SHEET_NAMES:
        scores[name] = load_eval_scores_from_sheet(SHEET_ID, name)
    

    texts = {
        "Boormachine advies_validation": BOORMACHINE_ADVICE_TEXT,
    }

    models = ["LL-01-pro"]
    main(texts, models, concepts['concepts'], output_dir, scores)

