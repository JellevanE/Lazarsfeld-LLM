from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from pprint import pprint
import json
from pathlib import Path
from colorama import init
from datetime import datetime
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from config.evaluation_config import EPSON_PRINTER_TEXT, BOL_TAFEL_TEXT
from config.config import MODEL, DEFAULT_WEIGHT, MODELS

from src.concepts import QuestionEval, DimensionEval, ConceptEval, ModelEval, TextEval, Concept, Dimension, Question
from src.update_concepts import process_concept_csv
from src.utils import fancy_print_output

from prompts.eval_prompt import sys_eval_prompt, base_eval_prompt
from prompts.voorbeelden import (
    B1,
    C1
    )

load_dotenv()
init(autoreset=True)  # Initialize colorama
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


# Function to evaluate a single question using LLM
def evaluate_question(
        client: OpenAI, 
        concept:Concept, 
        dimension:Dimension, 
        question_obj: Question, 
        input_text: str
        ) -> QuestionEval:

    # Extract question details
    question = question_obj["question"]
    positive_contribution = question_obj["positive_contribution"]
    examples = question_obj["examples"]

    system_prompt = sys_eval_prompt.format(concept=concept)
    base_prompt = base_eval_prompt.format(concept=concept, dimension=dimension, question=question, input_text=input_text, examples=examples)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": base_prompt}
        ],
        logprobs=True,
        top_logprobs=5
    )

    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    probability = None  # Initialize probability to None
    logprob_value = None  # Initialize logprob_value to None
    token = None

    for logprob in top_logprobs:
        token = logprob.token.lower()

        # If token is not true or false, assign None to probability
        if token != "true" and token != "false":
            continue

        if positive_contribution == True:
            if token == "true":
                token = logprob.token
                logprob_value = logprob.logprob
                probability = np.round(np.exp(logprob_value), 3)
                break # Exit loop after finding the first valid token

        elif positive_contribution == False:
            if token == "false":
                token = logprob.token
                logprob_value = logprob.logprob
                probability = 1 - np.round(np.exp(logprob_value), 3)
                break # Exit loop after finding the first valid token

    return {
        "label": question_obj["label"],
        "question": question,
        "answer": token,
        "score": probability,
        "logprob": logprob_value,
        "positive_contribution": positive_contribution
    }


# Function to evaluate a single dimension using LLM
def evaluate_dimension(
        client: OpenAI, 
        concept:Concept, 
        dimension: Dimension, 
        input_text: str
        ) -> DimensionEval:
    
    print(f"Evaluating dimension: {dimension['dimension_description']}")
    
    questions = dimension["questions"]
    question_scores = []

    for question in questions:
        question_eval = evaluate_question(client, concept, dimension, question, input_text)
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
        client: OpenAI,
        concept: Concept, 
        input_text: str
        ) -> ConceptEval:
    
    print(f"\nEvaluating concept: {concept['concept_description']}")
    
    dimensions = concept["dimensions"]
    dimension_scores = []

    for dimension in dimensions:
        dimension_eval = evaluate_dimension(client, concept, dimension, input_text)
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
        client: OpenAI, 
        concepts: list[Concept], 
        input_text: str
        ) -> ModelEval:
    
    print(f"\nEvaluating using model: {MODEL}")
    
    concept_scores = []
    for concept in concepts:
        concept_eval = evaluate_concept(client, concept, input_text)
        concept_scores.append(concept_eval)
    
    # Calculate overall score for the model, excluding None values
    scores = [c["overall_score"] for c in concept_scores if c["overall_score"] is not None]
    overall_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    return {
        "model_name": MODEL,
        "concepts_scores": concept_scores,
        "overall_score": overall_score,
        "weight": DEFAULT_WEIGHT
    }


def text_eval(
        client: OpenAI, 
        models:list, 
        text:str,
        label:str,
        concepts:list[Concept],
        ) -> TextEval:
    
    evaluations = {}
    for model in models:
        model_eval_result = model_eval(client, concepts, text)
        evaluations[model] = model_eval_result

    # Calculate aggregated score for the text, excluding None values
    scores = [m["overall_score"] for m in evaluations.values() if m["overall_score"] is not None]
    aggregated_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    metadata = {
        "models_used": models,
        "evaluation_parameters": {
            "system_prompt": str(sys_eval_prompt),
            "base_prompt": str(base_eval_prompt)
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


def main(texts:dict, models:list, concepts:list[Concept], output_dir) -> None:
    """Runs evaluation pipeline and saves results to JSON file."""
    # Initialize OpenAI client
    client = OpenAI()


    # Run evaluation
    for label, text in texts.items():
        text_eval_result = text_eval(client, models, text, label, concepts)
        # Save results to JSON file
        with open(f"{output_dir}{label}.json", "w") as f:
            json.dump(text_eval_result, f, indent=4)

    # Print results to console
    fancy_print_output(text_eval_result)


if __name__ == "__main__":
    # load concept from json file
    csv_path = Path('eval_concepts/LLM_eval_concepten - Taalniveau B1.csv')
    process_concept_csv(csv_path, output_filepath=Path('eval_concepts/taalniveau_b1_concept.json'), concept_name="Taalniveau_B1")

    with open ("eval_concepts/taalniveau_b1_concept.json", "r") as f:
        concepts = json.load(f)

    output_dir = "evaluation_results/"

    texts = {'epson_printer':EPSON_PRINTER_TEXT, 
             'bol_tafel':BOL_TAFEL_TEXT, 
             'b1_voorbeeld':B1, 
             'c1_voorbeeld':C1}
    models = ["gpt-3.5-turbo-0125", "gpt-4o", "gpt-4-turbo"]

    main(texts, models, concepts['concepts'], output_dir)



