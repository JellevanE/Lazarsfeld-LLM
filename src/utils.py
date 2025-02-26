from colorama import Fore, Style
from typing import Union, Dict, Any
from src.concepts import QuestionEval, DimensionEval, ConceptEval, ModelEval, TextEval

def color_score(score: float) -> str:
    """Returns color code based on score."""
    if score is None:
        return Style.DIM  # Greyed out for None
    if score < 0.4:
        return Fore.RED
    elif 0.4 <= score < 0.7:
        return Fore.YELLOW
    else:
        return Fore.GREEN

def fancy_print_output(eval_object: Union[Dict[str, Any], QuestionEval, DimensionEval, ConceptEval, ModelEval, TextEval], 
                       input_text: str = None, level: int = 0) -> None:
    """Prints evaluation results in a fancy format with colors, dynamically handling different evaluation object types."""

    indent = "\t" * level  # Indentation based on the level of nesting

    if "input_text" in eval_object and "evaluations" in eval_object:  # Check for TextEval
        eval_obj = eval_object # type: TextEval
        print("=====================================")
        print(Style.BRIGHT + "Evaluation Results" + Style.RESET_ALL)
        print("=====================================")
        print(f"Input Text: {eval_obj['input_text']}")
        for model_name, model_eval in eval_obj['evaluations'].items():
            fancy_print_output(model_eval, level=0)  # Recursive call for ModelEval

    elif "model_name" in eval_object and "concepts_scores" in eval_object:  # Check for ModelEval
        eval_obj = eval_object # type: ModelEval
        print(indent + Style.BRIGHT + f"Model: {eval_obj['model_name']}" + Style.RESET_ALL)
        score = eval_obj['overall_score']
        color = color_score(score)
        print(indent + Style.BRIGHT + "Overall Score: " + Style.RESET_ALL + f"{color}{score}{Style.RESET_ALL}\n")
        for concept_eval in eval_obj['concepts_scores']:
            fancy_print_output(concept_eval, level=level + 1)  # Recursive call for ConceptEval

    elif "concept_description" in eval_object and "dimensions" in eval_object:  # Check for ConceptEval
        eval_obj = eval_object # type: ConceptEval
        print(indent + Style.BRIGHT + f"Concept: {eval_obj['concept_description']}" + Style.RESET_ALL)
        score = eval_obj['overall_score']
        color = color_score(score)
        print(indent + Style.BRIGHT + "Overall Score: " + Style.RESET_ALL + f"{color}{score}{Style.RESET_ALL}")
        for dimension_eval in eval_obj['dimensions']:
            fancy_print_output(dimension_eval, level=level + 1)  # Recursive call for DimensionEval

    elif "dimension_description" in eval_object and "questions" in eval_object:  # Check for DimensionEval
        eval_obj = eval_object # type: DimensionEval
        print(indent + Style.BRIGHT + f"Dimension: {eval_obj['dimension_description']}" + Style.RESET_ALL)
        score = eval_obj['overall_score']
        color = color_score(score)
        print(indent + Style.BRIGHT + "Overall Score: " + Style.RESET_ALL + f"{color}{score}{Style.RESET_ALL}")
        for question_eval in eval_obj['questions']:
            fancy_print_output(question_eval, level=level + 1)  # Recursive call for QuestionEval

    elif "question" in eval_object and "answer" in eval_object:  # Check for QuestionEval
        eval_obj = eval_object # type: QuestionEval
        print(indent + f"Question: {eval_obj['question']}")
        print(indent + f"\tAnswer: {eval_obj['answer']}")
        score = eval_obj['score']
        color = color_score(score)
        print(indent + f"\tScore: {color}{score}{Style.RESET_ALL}")

    else:
        print("Unsupported evaluation object type.")