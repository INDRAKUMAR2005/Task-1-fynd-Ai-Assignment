import google.generativeai as genai
import pandas as pd
import json
import time
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error

API_KEY = "AIzaSyCX3jp99dBSNKge8pk1Fh62CltS0rSiKr8"
DATASET_PATH = "dataset/yelp.csv"
SAMPLE_SIZE = 5
SEED = 42

genai.configure(api_key=API_KEY)

def load_data():
    """Loads and samples the dataset."""
    try:
        df = pd.read_csv(DATASET_PATH)
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
        return df_sample
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_model():
    """Returns the Gemini model instance."""
    return genai.GenerativeModel("models/gemini-flash-latest")

# --- PROMPT DEFINITIONS ---

def get_prompt_v1_zeroshot(text):
    """
    Strategy 1: Zero-shot Baseline.
    Direct instruction to classify.
    """
    return f"""
    You are a sentiment analysis assistant.
    Analyze the following review and determine the star rating (1 to 5).
    Return the result strictly in JSON format.

    Review: "{text}"

    Output JSON format:
    {{
        "predicted_stars": <int>,
        "explanation": "<string>"
    }}
    """

def get_prompt_v2_fewshot(text):
    """
    Strategy 2: Few-shot.
    Providing examples for context.
    """
    return f"""
    Determine the star rating (1-5) for the review based on the text.
    Return JSON format: {{ "predicted_stars": <int>, "explanation": "<string>" }}

    Examples:
    Review: "The food was cold and the service was terrible."
    Output: {{ "predicted_stars": 1, "explanation": "Negative sentiment regarding both food quality and service." }}

    Review: "It was okay. Not great, but not bad either."
    Output: {{ "predicted_stars": 3, "explanation": "Neutral sentiment, expressing mediocrity." }}

    Review: "Absolutely amazing experience! The staff was friendly and the steak was perfect."
    Output: {{ "predicted_stars": 5, "explanation": "Highly positive feedback on service and food." }}

    Review: "{text}"
    Output:
    """

def get_prompt_v3_roleplay(text):
    """
    Strategy 3: Role-based + Chain-of-Thought (implied in explanation).
    Acting as an expert critic.
    """
    return f"""
    You are an expert food critic and sentiment analyst. Your task is to accurately predict the star rating (1-5) of a Yelp review.
    
    Analyze the review carefully:
    1. Identify key sentiment words (positive/negative).
    2. Assess the intensity of the emotions.
    3. Look for specific complaints or praises (food, service, ambiance).
    4. Based on this analysis, assign a score.

    Review: "{text}"

    Provide your response as a valid JSON object.
    {{
        "predicted_stars": <int>,
        "explanation": "<brief reasoning based on analysis steps>"
    }}
    """

PROMPTS = {
    "v1_zeroshot": get_prompt_v1_zeroshot,
    "v2_fewshot": get_prompt_v2_fewshot,
    "v3_roleplay": get_prompt_v3_roleplay
}

def call_api(model, prompt):
    """Calls the Gemini API with error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt
            )
            return response.text
        except Exception as e:
            print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1)) # Exponential backoff
    return None

def parse_response(response_text):
    """Parses JSON from response."""
    try:
        # Gemini usually returns strictly JSON with response_mime_type, but safety strip
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3]
            
        data = json.loads(clean_text)
        return int(data.get("predicted_stars")), data.get("explanation")
    except Exception as e:
        # print(f"JSON Parsing Error: {e}, Text: {response_text}")
        return None, "Parsing Error"

def evaluate_strategy(name, prompt_func, df, model):
    print(f"--- Running Evaluation for: {name} ---")
    predictions = []
    valid_json_count = 0
    
    # Using tqdm for progress bar
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        prompt = prompt_func(text)
        
        response_text = call_api(model, prompt)
        
        if response_text:
            stars, explanation = parse_response(response_text)
            if stars is not None:
                valid_json_count += 1
                predictions.append(stars)
            else:
                predictions.append(-1) # Mark as invalid
        else:
            predictions.append(-1)
            
    df[f'{name}_pred'] = predictions
    
    # Filter valid predictions for accuracy metrics
    valid_mask = df[f'{name}_pred'] != -1
    valid_df = df[valid_mask]
    
    accuracy = accuracy_score(valid_df['stars'], valid_df[f'{name}_pred']) if not valid_df.empty else 0
    validity_rate = valid_json_count / len(df)
    
    results = {
        "Strategy": name,
        "Accuracy": round(accuracy, 4),
        "JSON_Validity": round(validity_rate, 4),
        "Valid_Samples": len(valid_df)
    }
    
    return results

def main():
    print("Loading data...")
    df = load_data()
    if df is None:
        return

    print(f"Data loaded. Sample size: {len(df)}")
    model = get_model()
    
    all_results = []
    
    for name, prompt_func in PROMPTS.items():
        res = evaluate_strategy(name, prompt_func, df, model)
        all_results.append(res)
        
    print("\n--- Final Results ---")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_markdown(index=False))
    
    # Save detailed results
    df.to_csv("evaluation_results_detailed.csv", index=False)
    results_df.to_csv("evaluation_metrics.csv", index=False)
    
    # Generate Report
    with open("report.md", "w") as f:
        f.write("# Yelp Rating Prediction Report\n\n")
        f.write("## Comparison Table\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Discussion\n")
        f.write("### 1. Zero-shot Baseline\n")
        f.write("Direct asking. Good baseline, but might miss nuances or strict formatting occasionally.\n\n")
        f.write("### 2. Few-shot\n")
        f.write("Provided 3 examples. Typically stabilizes the output format and helps with ambiguous cases.\n\n")
        f.write("### 3. Role-based + CoT\n")
        f.write("Asked to act as a critic and analyze steps. This generally improves reasoning for complex reviews where sentiment doesn't match the score trivially.\n")

if __name__ == "__main__":
    main()

