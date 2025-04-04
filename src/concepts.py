from typing import TypedDict, List, Dict, Any, Optional

class Question(TypedDict):
    label: str
    question: str
    positive_contribution: bool  # if True, higher score is better; if False, lower score is better
    examples: str

class Dimension(TypedDict):
    dimension_description: str
    questions: List[str]
    weight: float  # importance weight within parent concept (default 1.0)

class Concept(TypedDict):
    concept_description: str
    dimensions: List[str]
    weight: float  # importance weight within overall evaluation (default 1.0)

class QuestionEval(TypedDict):
    label: str
    question: str
    answer: str
    score: float  # percentage score from 0 to 1
    logprob: Optional[float]  # log probability of the ansœwer
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
    concepts_scores: List[ConceptEval]
    overall_score: float
    weight: float # importance weight within overall evaluation (default 1.0)
    # top_logprobs: List[Dict[str, float]]  # logprobs for the model's responses

class TextEval(TypedDict):
    label: str
    input_text: str
    concepts: List[Concept]
    evaluations: Dict[str, ModelEval]  # model_id -> ModelEval mapping for multiple model support
   # aggregated_scores: dict{str:float}  # combined score across all models used
    metadata: Dict[str, Any]  # information about evaluation parameters, etc.
    timestamp: str  # when evaluation was performed


class ValidationScores(TypedDict):
    sheet_name: str
    scores: dict[str:int]
    label: str
    text: str