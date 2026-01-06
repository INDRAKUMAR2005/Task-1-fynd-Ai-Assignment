# Yelp Rating Prediction via Prompting (Task 1)

This project implements **prompt-based rating prediction** for Yelp reviews using **Google’s Gemini LLM**.  
The objective is to classify reviews into **1–5 star ratings** and analyze how different **prompting strategies** impact model performance.

This work is submitted as **Task 1** for the **Fynd AI Intern – Take Home Assessment 2.0**.

---

## Overview

We design and evaluate **three prompting approaches** for structured sentiment classification:

1. **Zero-shot Prompting**  
   Directly asks the model to predict a star rating without examples.

2. **Few-shot Prompting**  
   Provides representative examples for selected star ratings to guide the model.

3. **Role-based Prompting**  
   Instructs the model to act as a food critic and reason about sentiment before classifying.

Each approach returns a **strict JSON response** in the following format:

json
{
  "predicted_stars": 4,
  "explanation": "Brief reasoning for the assigned rating."
}

Dataset

Source: Yelp Reviews Dataset (Kaggle)

Location: dataset/yelp.csv

Due to Gemini free-tier API rate limits, a small sampled subset of the dataset is used for evaluation.
While the task recommends ~200 samples, this experiment focuses on relative prompt comparison rather than absolute accuracy.

Prerequisites

Python 3.9+

Google Gemini API Key

Installation

Install the required Python packages:

pip install google-generativeai pandas tqdm tabulate scikit-learn

Configuration

The main script classify_reviews.py can be configured using the following variables:

SAMPLE_SIZE – Number of reviews processed

API_KEY – Google Gemini API key

DATASET_PATH – Path to the Yelp CSV file

MODEL_NAME – Gemini model used (default: gemini-flash-latest)

Usage

Run the classification pipeline:

python classify_reviews.py


The script will:

Execute all three prompting strategies

Validate and parse structured JSON responses

Compare predictions with ground-truth ratings

Aggregate evaluation metrics

Evaluation Metrics

For each prompting strategy, the following metrics are reported:

Accuracy – Predicted vs actual star ratings

JSON Validity Rate – Percentage of valid structured outputs

Valid Samples – Successfully parsed responses

A comparison table and qualitative analysis are provided in report.md.

Outputs

Running the script generates:

evaluation_results_detailed.csv
→ Per-review predictions and metadata

evaluation_metrics.csv
→ Aggregated metrics per prompting strategy

report.md
→ Detailed discussion covering:

Prompt design and iterations

Quantitative results

Reliability observations

Trade-offs and limitations

Notes & Limitations

Few-shot prompting increases token usage and is more sensitive to API rate limits.

Smaller sample sizes reduce statistical confidence but still allow meaningful prompt-level comparison.

All LLM interactions are performed server-side via Python, with explicit JSON validation.

Author

Indra Kumar
Fynd AI Intern – Take Home Assessment 2.0


