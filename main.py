from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from pprint import pprint
from typing import TypedDict, List, Dict, Any
import json

from config.evaluation_config import SAMPLE_EVALUATION_CONFIG

from config.config import MODEL, DEFAULT_WEIGHT

load_dotenv()

class QuestionEval(TypedDict):
    question: str
    answer: str
    score: float  # percentage score from 0 to 1
    positive_contribution: bool  # if True, higher score is better; if False, lower score is better

class DimensionEval(TypedDict):
    dimension_description: str
    questions: List[QuestionEval]
    overall_score: float # weighted average of question scores
    weight: float  # importance weight within parent concept (default 1.0)

class ConceptEval(TypedDict):
    concept_description: str
    dimensions: List[DimensionEval]
    overall_score: float # weighted average of dimension scores
    weight: float  # importance weight within overall evaluation (default 1.0)

class ModelEval(TypedDict):
    model_name: str
    concepts: List[ConceptEval]
    overall_score: float
    weight: float # importance weight within overall evaluation (default 1.0)
    # top_logprobs: List[Dict[str, float]]  # logprobs for the model's responses

class TextEval(TypedDict):
    input_text: str
    evaluations: Dict[str, ModelEval]  # model_id -> ModelEval mapping for multiple model support
    aggregated_score: float  # combined score across all models used
    metadata: Dict[str, Any]  # information about evaluation parameters, etc.
    timestamp: str  # when evaluation was performed


# Function to evaluate a single question using LLM
def evaluate_question(client: OpenAI, system_prompt:str, base_prompt:str, 
                    question: str, input_text: str, 
                    positive_contribution: bool = True
                    ) -> QuestionEval:
    
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": base_prompt + input_text + question}
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
                probability = np.exp(logprob_value)

                # print(f"Token: {token}")
                # print(f"Logprob: {logprob_value}")
                # print(f"Probability: {probability}")
                break # Exit loop after finding the first valid token

        elif positive_contribution == False:
            if token == "false":
                token = logprob.token
                logprob_value = logprob.logprob
                probability = np.exp(logprob_value)

                # print(f"Token: {token}")
                # print(f"Logprob: {logprob_value}")
                # print(f"Probability: {probability}")
                break # Exit loop after finding the first valid token

    return {
        "question": question,
        "answer": token,
        "score": probability,
        "positive_contribution": positive_contribution
    }


# Function to evaluate a single dimension using LLM
def evaluate_dimension(client: OpenAI, system_prompt:str, base_prompt:str, 
                        dimension: DimensionEval, input_text: str) -> DimensionEval:
    
    questions = dimension["questions"]
    question_scores = []

    for question in questions:
        question_eval = evaluate_question(client, system_prompt, base_prompt, 
                                           question["question"], input_text, 
                                           question["positive_contribution"])
        question_scores.append(question_eval)

    # Calculate overall score for the dimension, excluding None values
    scores = [q["score"] for q in question_scores if q["score"] is not None]
    overall_score = np.mean(scores) if scores else None  # Default to 0 if all scores are None


    return {
        "dimension_description": dimension["dimension_description"],
        "questions": question_scores,
        "overall_score": overall_score,
        "weight": dimension.get("weight", DEFAULT_WEIGHT)
    }


# Function to evaluate a single concept using LLM
def evaluate_concept(client: OpenAI, system_prompt:str, base_prompt:str, 
                        concept: ConceptEval, input_text: str) -> ConceptEval:
    
    dimensions = concept["dimensions"]
    dimension_scores = []

    for dimension in dimensions:
        dimension_eval = evaluate_dimension(client, system_prompt, base_prompt, 
                                             dimension, input_text)
        dimension_scores.append(dimension_eval)

    # Calculate overall score for the concept, excluding None values
    scores = [d["overall_score"] for d in dimension_scores if d["overall_score"] is not None]
    overall_score = np.mean(scores) if scores else None  # Default to 0 if all scores are None

    return {
        "concept_description": concept["concept_description"],
        "dimensions": dimension_scores,
        "overall_score": overall_score,
        "weight": concept.get("weight", DEFAULT_WEIGHT)
    }


if __name__ == "__main__":
    # Initialize OpenAI client
    test_concepts = []
    client = OpenAI()
    for concept in SAMPLE_EVALUATION_CONFIG['concepts']:
        concept_eval = evaluate_concept(client, 
                                        system_prompt="You are a helpful assistant.", 
                                        base_prompt="Evaluate the following text: ",
                                        concept=concept,
                                        input_text="Deze tekst is helemaal ruk, maar hopelijk wel super duidelijk. Ennnnn je moeder.")
        test_concepts.append(concept_eval)
    pprint(test_concepts)

    # for dimension in SAMPLE_EVALUATION_CONFIG['concepts'][0]['dimensions']:
    #     dimension_eval = evaluate_dimension(client, 
    #                                         system_prompt="You are a helpful assistant.", 
    #                                         base_prompt="Evaluate the following text: ",
    #                                         dimension=dimension,
    #                                         input_text="This is a sample text to evaluate.")
    #    pprint(dimension_eval)


