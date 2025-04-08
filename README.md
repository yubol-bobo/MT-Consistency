# Firm or Fickle? Evaluating LLM Consistency

This repository accompanies our paper [**"Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions"**](arxiv.org/abs/2503.22353) by Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, and Rema Padman (Carnegie Mellon University). The work introduces a systematic evaluation framework for assessing the consistency of large language models (LLMs) over multi-turn interactions. It also proposes a novel Position-Weighted Consistency (PWC) score and a Confidence-Aware Response Generation (CARG) framework for robust performance in high-stakes domains.

---

## Abstract

LLMs have shown impressive performance on a variety of tasks but can struggle with maintaining consistent responses throughout sequential interactions. Our contributions include:

- The **Position-Weighted Consistency (PWC) score** to evaluate early-stage response stability.
- The **Confidence-Aware Response Generation (CARG)** framework to dynamically incorporate model confidence in the generation process.
- A curated benchmark and detailed experiments demonstrating key differences in LLM consistency.

## Repository Structure

- **Data/**  
  Stores input data files needed for experiments.  
  - `cleaned_data.csv` – Example of a preprocessed dataset used in experiments or demonstrations.

- **Outputs/**  
  Stores model outputs, logs, or benchmarking results.  
  - **pwc_benchmark/** – Contains outputs or results files (e.g., JSON, CSV, plots) related to measuring Position-Weighted Consistency (PWC) or other metrics.

- **src/**  
  Source code for running experiments and utilities:
  - `__init__.py` – Makes the `src/` directory a Python package.
  - `chat_module.py` – Contains functions or classes that handle multi-turn chat logic, prompt generation, or model interaction.
  - `config.py` – Centralized configuration (hyperparameters, file paths, environment variables).  
  - `data_utils.py` – Helper functions for loading, cleaning, or preprocessing data.  
  - `experiment.py` – Main script or module that orchestrates experiment workflows (e.g., running multiple trials, computing metrics).

- **.env**  
  Environment variables file (optional). Place credentials or private environment variables here (e.g., API keys, logging configs) and do **not** commit sensitive info to version control.

- **.gitignore**  
  Specifies intentionally untracked files and folders (e.g., `.env`, logs, cache files).

- **LICENSE**  
  The repository’s license.

- **main.py**  
  A main entry point to run or demo the code, depending on your workflow. It may consolidate various steps (data loading, config parsing, experiment execution).

- **README.md**  
  This file, providing an overview and usage instructions.

- **requirements.txt**  
  Lists Python dependencies. Install them via:
  ```bash
  pip install -r requirements.txt
