import os
from src.config import load_api_keys, SAVE_PATH
from src.data_utils import load_data, convert_to_qa_pairs
from src.experiment import run_experiment_rep, run_experiment_diverse
from tqdm import tqdm
import time
# from google.genai.errors import ServerError


def main():
    # Load API keys (ensure they are set in your environment)
    api_keys = load_api_keys()

    # Path to your CSV data file (adjust the path as needed)
    csv_path = "./data/health_data_converted.csv"
    data_df = load_data(csv_path)[:700]
    qa_pairs = convert_to_qa_pairs(data_df)

    # Define model list and experiment parameters
    model_list = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "claude-3-5-sonnet-latest",
        "gpt-4o",
        "mistral-large-latest",
        "deepseek-chat",
        "qwen-max-latest"
    ]

    # Set the experiment type:
    # Choose either 'diverse' or 'repetitive'
    exp = 'diverse'  # For diverse experiment, change to 'repetitive' as needed
    batch_size = 100
    for batch in tqdm(range(1, 8)):  # 1 to 7 inclusive
        qa_batch = qa_pairs[batch_size * (batch - 1): batch_size * batch]
        print(f"Running batch {batch}...")
        model = model_list[6]
        rounds = 1
        random_order = False

        # Define the output directory based on experiment type
        if exp == 'diverse':
            experiment_save_path = os.path.join(SAVE_PATH, "carg")
        elif exp == 'repetitive':
            experiment_save_path = os.path.join(SAVE_PATH, "repetitive")
        else:
            raise ValueError("Invalid experiment type specified. Use 'diverse' or 'repetitive'.")

        # Modified: Save to health/carg/gpt directory structure
        experiment_save_path = os.path.join("Outputs", "health", "diverse", "carg")
            
        # Create the output directory if it doesn't exist
        if not os.path.exists(experiment_save_path):
            os.makedirs(experiment_save_path)

        # Run the appropriate experiment function based on the experiment type
        if exp == 'diverse':
            print('Diverse Experiment Running...')
            run_experiment_diverse(
                qa_pairs=qa_batch,
                model=model,
                rounds=rounds,
                random_order=random_order,
                batch_idx=batch,        # Pass batch as batch_idx for diverse experiments
                save_path=experiment_save_path
            )
        elif exp == 'repetitive':
            print('Repetitive Experiment Running...')
            run_experiment_rep(
                qa_pairs=qa_batch,
                model=model,
                rounds=rounds,
                random_order=random_order,
                batch=batch,
                save_path=experiment_save_path
            )

if __name__ == '__main__':
    main()