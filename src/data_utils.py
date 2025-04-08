import ast
import json
import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data from the given path.
    """
    df = pd.read_csv(csv_path)
    return df

def convert_to_qa_pairs(df: pd.DataFrame) -> list:
    """
    Convert DataFrame to list of QA pair dictionaries.
    Each dictionary contains:
      - question: string
      - choices: list
      - answer: string
    """
    qa_pairs = []
    for _, row in df.iterrows():
        # Safely convert string representation of list to an actual list
        choices_list = ast.literal_eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
        qa_pair = {
            "question": row['question'],
            "choices": choices_list,
            "answer": row['answer']
        }
        qa_pairs.append(qa_pair)
    # Optionally print a sample for verification
    print("Sample QA Pair:")
    print(json.dumps(qa_pairs[0], indent=4))
    return qa_pairs
