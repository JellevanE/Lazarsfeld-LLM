# Lazarsfeld-LLM

A framework for evaluating LLM output using social science methodologies, inspired by the work of sociologist Paul Lazarsfeld.

## Overview

Lazarsfeld-LLM is a prototype system that evaluates the quality of text outputs from Large Language Models (LLMs) using a structured, hierarchical approach similar to social science research methodologies. The system breaks down complex evaluation criteria into measurable dimensions and questions, allowing for systematic assessment of text quality.

The framework is designed to:

1. Evaluate text outputs against predefined concepts (e.g., B1 language level)
2. Support multiple evaluation dimensions within each concept
3. Provide detailed scoring at question, dimension, and concept levels
4. Compare outputs across different LLM models
5. Generate comprehensive evaluation reports

## Key Features

- **Hierarchical Evaluation Structure**: Concepts → Dimensions → Questions
- **Flexible Concept Definitions**: Define custom evaluation criteria through JSON configuration
- **Multi-Model Comparison**: Compare output quality across different LLM models
- **Detailed Scoring**: Get scores at all levels with aggregated results
- **Colorized Console Output**: Visualize evaluation results with intuitive color coding
- **Evaluation Caching**: Cache evaluation results to improve performance

## Project Structure

- `main.py`: Core evaluation pipeline
- `src/`: Source code for the evaluation framework
  - `concepts.py`: Type definitions for evaluation structure
  - `utils.py`: Utility functions for output formatting
  - `analysis.py`: Tools for analyzing evaluation results
- `config/`: Configuration files
  - `config.py`: Default configuration parameters
  - `evaluation_config.py`: Sample text configuration
- `prompts/`: Prompt templates for LLM-based evaluation
  - `eval_prompt.py`: Evaluation prompt templates
  - `voorbeelden.py`: Example texts for evaluation
- `eval_concepts/`: Predefined evaluation concepts
  - `taalniveau_b1_concept.json`: B1 language level evaluation concept

## How It Works

1. **Define Evaluation Concepts**: Create a JSON file with your evaluation criteria structured as concepts, dimensions, and questions.
2. **Configure Input Text**: Provide the text you want to evaluate.
3. **Run Evaluation**: The system sends prompts to LLMs to evaluate the text against your defined criteria.
4. **Review Results**: Get detailed scores and insights at all levels of the evaluation hierarchy.

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Lazarsfeld-LLM.git
   cd Lazarsfeld-LLM
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

1. Define your evaluation concept in a JSON file (see `eval_concepts/taalniveau_b1_concept.json` for an example).

2. Run the evaluation:
   ```python
   python main.py
   ```

3. Or import the evaluation functions in your own code:
   ```python
   from main import evaluate_text
   
   results = evaluate_text(
       text="Your text to evaluate",
       concepts=your_concepts,
       models=["gpt-4o", "gpt-3.5-turbo"]
   )
   ```

## Examples

The project includes sample evaluation concepts and example texts:

- B1 language level evaluation
- Text samples for printer manuals and product descriptions

## Customization

You can create your own evaluation concepts by following the structure in `taalniveau_b1_concept.json`. Each concept consists of:

- Dimensions: Categories of evaluation
- Questions: Specific criteria within each dimension
- Weights: Importance factors for scoring

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is inspired by the work of sociologist Paul Lazarsfeld, who pioneered methods for measuring complex social phenomena through structured approaches.
