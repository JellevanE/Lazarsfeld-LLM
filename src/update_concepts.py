import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, DefaultDict
from src.concepts import Concept, Dimension, Question


def load_concept_data(csv_filepath: Path) -> pd.DataFrame:
    """Load concept data from a CSV file."""
    try:
        df = pd.read_csv(csv_filepath)
        print(f"Successfully loaded data from {csv_filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_questions_from_df(df: pd.DataFrame) -> Tuple[Dict[str, List[Question]], List[str]]:
    """Create Question objects from DataFrame rows and group them by dimension."""
    questions_by_dimension: Dict[str, List[Question]] = defaultdict(list)
    
    # Process each row and create Question objects
    for _, row in df.iterrows():
        question_obj = Question(
            label=row['Label'],
            question=row['Question'],
            positive_contribution=row['positive_contribution'],
            examples=str(row['Examples']) if not pd.isna(row['Examples']) else ""
        )
        
        # Group questions by their dimension
        questions_by_dimension[row['Dimension']].append(question_obj)
    
    # Get unique dimensions
    unique_dimensions = list(df['Dimension'].unique())
    print(f"Created {len(df)} questions across {len(unique_dimensions)} dimensions")
    
    return questions_by_dimension, unique_dimensions


def create_dimensions(
    questions_by_dimension: Dict[str, List[Question]], 
    unique_dimensions: List[str]
) -> List[Dimension]:
    """Create Dimension objects with weights based on question counts."""
    dimensions: List[Dimension] = []
    total_questions = sum(len(questions) for questions in questions_by_dimension.values())
    
    # Create a Dimension object for each unique dimension
    for dim_name in unique_dimensions:
        dim_questions = questions_by_dimension[dim_name]
        # Calculate weight based on number of questions
        weight = len(dim_questions) / total_questions
        
        # Create dimension object with list of question labels
        dimension = Dimension(
            dimension_description=dim_name,
            questions=[q["label"] for q in dim_questions],
            weight=weight
        )
        dimensions.append(dimension)
    
    print(f"Created {len(dimensions)} dimensions with weighted importance")
    return dimensions


def create_concept(dimensions: List[Dimension], concept_name: str) -> Concept:
    """Create a Concept object containing a list of dimensions."""
    concept = Concept(
        concept_description=concept_name,
        dimensions=[d["dimension_description"] for d in dimensions],  # List of dimension descriptions
        weight=1.0  # Single concept gets full weight
    )
    
    print(f"Created concept '{concept_name}' with {len(dimensions)} dimensions")
    return concept


def build_output_structure(
    concept: Concept, 
    dimensions: List[Dimension], 
    questions_by_dimension: Dict[str, List[Question]]
) -> Dict[str, Any]:
    """Build the complete structure for saving with proper nesting of objects."""
    # Create a list to hold nested dimensions with their questions
    nested_dimensions = []
    
    for dimension in dimensions:
        dim_name = dimension["dimension_description"]
        
        # Get the questions for this dimension
        dim_questions = questions_by_dimension[dim_name]
        
        # Create a new dimension object with nested questions
        nested_dimension = {
            "dimension_description": dim_name,
            "questions": dim_questions,  # List of complete question objects
            "weight": dimension["weight"]
        }
        nested_dimensions.append(nested_dimension)
    
    # Create a new concept object with nested dimension objects
    nested_concept = {
        "concept_description": concept["concept_description"],
        "dimensions": nested_dimensions,  # List of dimension objects with nested questions
        "weight": concept["weight"]
    }
    
    # Put the concept into a list inside the concepts key
    output_data = {
        "concepts": [nested_concept]  # List of concept objects
    }
    
    return output_data


def save_to_json(data: Dict[str, Any], output_path: Path) -> Path:
    """Save data structure to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Concept structure saved to {output_path}")
    return output_path


def process_concept_csv(
    csv_filepath: Path, 
    output_filepath: Optional[Path] = None, 
    concept_name: Optional[str] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Main function to process a concept CSV file and generate a structured concept.
    """
    # Convert string path to Path object if needed
    if isinstance(csv_filepath, str):
        csv_filepath = Path(csv_filepath)
    
    # Default output filepath if none provided
    if output_filepath is None:
        base_name = csv_filepath.stem
        output_filepath = Path('eval_concepts') / f"{base_name.lower().replace(' ', '_')}_concept.json"
    elif isinstance(output_filepath, str):
        output_filepath = Path(output_filepath)
    
    # Default concept name if none provided
    if concept_name is None:
        concept_name = csv_filepath.stem
    
    # Load and process the data
    df = load_concept_data(csv_filepath)
    questions_by_dimension, unique_dimensions = create_questions_from_df(df)
    dimensions = create_dimensions(questions_by_dimension, unique_dimensions)
    concept = create_concept(dimensions, concept_name)
    
    # Build and save the output
    output_data = build_output_structure(concept, dimensions, questions_by_dimension)
    save_to_json(output_data, output_filepath)
    
    return output_data, output_filepath


if __name__ == "__main__":
    # Example usage when script is run directly
    csv_path = Path('eval_concepts/LLM_eval_concepten - Taalniveau B1.csv')
    process_concept_csv(csv_path, output_filepath=Path('eval_concepts/taalniveau_b1_concept.json'), concept_name="Taalniveau_B1")


