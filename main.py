from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from pprint import pprint
import json
from colorama import init
from datetime import datetime


from config.evaluation_config import SAMPLE_EVALUATION_CONFIG, EPSON_PRINTER_TEXT, BOL_TAFEL_TEXT
from config.config import MODEL, DEFAULT_WEIGHT, MODELS

from src.concepts import QuestionEval, DimensionEval, ConceptEval, ModelEval, TextEval, Concept, Dimension, Question
from src.utils import fancy_print_output

load_dotenv()
init(autoreset=True)  # Initialize colorama

# Function to evaluate a single question using LLM
def evaluate_question(client: OpenAI, system_prompt:str, base_prompt:str, 
                    question_obj: Question, input_text: str
                    ) -> QuestionEval:

    # Extract question details
    question = question_obj["question"]
    positive_contribution = question_obj["positive_contribution"]
    examples = question_obj["examples"]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": base_prompt + input_text + question + f"Examples: {examples}"}
        ],
        logprobs=True,
        top_logprobs=5
    )

    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    probability = None  # Initialize probability to None

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
        "question": question,
        "answer": token,
        "score": probability,
        "positive_contribution": positive_contribution
    }


# Function to evaluate a single dimension using LLM
def evaluate_dimension(client: OpenAI, system_prompt:str, base_prompt:str, 
                        dimension: Dimension, input_text: str) -> DimensionEval:
    
    print(f"Evaluating dimension: {dimension['dimension_description']}")
    
    questions = dimension["questions"]
    question_scores = []

    for question in questions:
        question_eval = evaluate_question(client, system_prompt, base_prompt, 
                                           question, input_text)
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
def evaluate_concept(client: OpenAI, system_prompt:str, base_prompt:str, 
                        concept: Concept, input_text: str) -> ConceptEval:
    
    print(f"\nEvaluating concept: {concept['concept_description']}")
    
    dimensions = concept["dimensions"]
    dimension_scores = []

    for dimension in dimensions:
        dimension_eval = evaluate_dimension(client, system_prompt, base_prompt, 
                                             dimension, input_text)
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


def model_eval(client: OpenAI, system_prompt:str, base_prompt:str,
               concepts: list[Concept], input_text: str) -> ModelEval:
    
    print(f"\nEvaluating using model: {MODEL}")
    
    concept_scores = []
    for concept in concepts:
        concept_eval = evaluate_concept(client, system_prompt, base_prompt, 
                                        concept, input_text)
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


def text_eval(client: OpenAI, models:list, text:str,
            concepts:list[Concept], system_prompt:str,
            base_prompt:str) -> TextEval:
    
    evaluations = {}
    for model in models:
        model_eval_result = model_eval(client, system_prompt, base_prompt, 
                                        concepts, text)
        evaluations[model] = model_eval_result

    # Calculate aggregated score for the text, excluding None values
    scores = [m["overall_score"] for m in evaluations.values() if m["overall_score"] is not None]
    aggregated_score = np.round(np.mean(scores) if scores else None, 3)  # Default to 0 if all scores are None

    metadata = {
        "models_used": models,
        "evaluation_parameters": {
            "system_prompt": system_prompt,
            "base_prompt": base_prompt
        }
    }

    timestamp = datetime.now().strftime("%Y-%m-%d")


    return {
        "input_text": text,
        "evaluations": evaluations,
        "aggregated_score": aggregated_score,
        "metadata": metadata,
        "timestamp": timestamp
    }

    

def main(text:str, models:list, concepts:list[Concept], output_file) -> None:
    """Runs evaluation pipeline and saves results to JSON file."""
    # Initialize OpenAI client
    client = OpenAI()

    # Define system and base prompts
    system_prompt = "You are a helpful assistant. You can only respond with True or False."
    base_prompt = "Evalueer de volgende tekst en beantwoorde de vraag met True of False. Tekst: "

    # Run evaluation
    text_eval_result = text_eval(client, models, text, concepts, system_prompt, base_prompt)

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(text_eval_result, f, indent=4)

    # Print results to console
    fancy_print_output(text_eval_result)

    


if __name__ == "__main__":
    # load concept from json file
    with open ("eval_concepts/taalniveau_b1_concept.json", "r") as f:
        concepts = json.load(f)

    output_file = "evaluation_results/B1_bol_test.json"
    # Run main function
    main(BOL_TAFEL_TEXT, MODELS, concepts['concepts'], output_file)



