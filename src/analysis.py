import numpy as np
import pandas as pd
import json
import matplotlib as plt
from concepts import TextEval

# load json data in TextEval format
with open('evaluation_results/new_prompt_test.json', 'r') as f:
    eval_data = TextEval(json.load(f))

print(eval_data['concepts'])

print(eval_data['input_text'])






