import os
from src.config import load_api_keys, SAVE_PATH
from src.data_utils import load_data, convert_to_qa_pairs
from src.experiment import run_experiment_rep

def main():
    # Load API keys (ensure they are set in your environment)
    api_keys = load_api_keys()

    # Path to your CSV data file (adjust the path as needed)
    csv_path = "./Data/cleaned_data.csv"
    data_df = load_data(csv_path)
    qa_pairs = convert_to_qa_pairs(data_df)

    # Define model and experiment parameters
    model_list = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "gemini-2.0-flash-exp",
        "claude-3-5-sonnet-latest",
        "gpt-4o",
        "mistral-large-latest",
        "deepseek-reasoner",
        "qwen-max-latest"
    ]
    # Example: Selecting one model and setting batch parameters
    batch = 1
    # Slice the qa_pairs list based on batch (example: 100 items per batch)
    batch_size = 6
    qa_batch = qa_pairs[batch_size * (batch-1): batch_size * batch]
    model = model_list[1]
    rounds = 8
    random_order = False

    # Define the directory where experiment results will be saved
    experiment_save_path = os.path.join(SAVE_PATH, "exp_2")
    if not os.path.exists(experiment_save_path):
        os.makedirs(experiment_save_path)

    # Run repetitive experiment
    run_experiment_rep(qa_pairs=qa_batch, model=model, rounds=rounds, random_order=random_order, batch=batch, save_path=experiment_save_path)

if __name__ == '__main__':
    main()
