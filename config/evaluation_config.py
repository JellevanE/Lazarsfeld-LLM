"""
Configuration file for text evaluation framework.
Define concepts, dimensions, and questions for evaluation.
"""

SAMPLE_EVALUATION_CONFIG = {
    "concepts": [
        {
            "concept_description": "Clarity of Expression",
            "weight": 1.0,
            "dimensions": [
                {
                    "dimension_description": "Sentence Structure",
                    "weight": 0.7,
                    "questions": [
                        {
                            "question": "Are the sentences in this text well-structured with clear subjects and verbs? Answer with True or False.",
                            "positive_contribution": True
                        },
                        {
                            "question": "Does the text contain run-on sentences that are difficult to follow? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                },
                {
                    "dimension_description": "Vocabulary Usage",
                    "weight": 0.5,
                    "questions": [
                        {
                            "question": "Is the vocabulary used in this text appropriate for the intended audience?",
                            "positive_contribution": True
                        },
                        {
                            "question": "Does the text use unnecessarily complex terminology? Answer with True or False.",
                            "positive_contribution": False
                        },
                        {
                            "question": "Do you find the text humorous? And do you think it was written by a machine? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                }
            ]
        },
        {
            "concept_description": "Logical Coherence",
            "weight": 1.2,
            "dimensions": [
                {
                    "dimension_description": "Argument Structure",
                    "weight": 1.0,
                    "questions": [
                        {
                            "question": "Does the text present a clear logical progression of ideas?",
                            "positive_contribution": True
                        },
                        {
                            "question": "Are there logical fallacies or contradictions in the text? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                },
                {
                    "dimension_description": "Evidence Support",
                    "weight": 0.8,
                    "questions": [
                        {
                            "question": "Are claims in the text supported by appropriate evidence? Answer with True or False.",
                            "positive_contribution": True
                        },
                        {
                            "question": "Does the text make unsupported assertions? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                }
            ]
        }
    ],
    "reference_texts": {
        "high_quality": "This is a sample high-quality text that would score well...",
        "medium_quality": "This is a sample medium-quality text with some issues...",
        "low_quality": "This is a poorly written sample with many problems..."
    },
    "models": [
        {
            "model_id": "gpt-4",
            "weight": 0.7,  # weight in final aggregated score
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 50
            }
        },
        {
            "model_id": "gpt-3.5-turbo",
            "weight": 0.3,
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 50
            }
        }
    ],
    "aggregation_method": "weighted_average"  # how to combine scores from different models
}
