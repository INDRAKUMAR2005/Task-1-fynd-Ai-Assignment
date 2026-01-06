# Yelp Rating Prediction via Prompting

This project classifies Yelp reviews into 1-5 star ratings using Google's Gemini API and compares three different prompting strategies.

## Prerequisites

- Python 3.9+
- A Google Gemini API Key (configured in `classify_reviews.py`)

## Installation

1.  Clone or download this repository.
2.  Install the required Python packages:

    ```bash
    pip install google-generativeai pandas tqdm tabulate scikit-learn
    ```

## Usage

Run the main classification script:

```bash
python classify_reviews.py
```

## Configuration

You can modify `classify_reviews.py` to change:
- `SAMPLE_SIZE`: Number of reviews to process (Default is set to 5 to avoid rate limits).
- `API_KEY`: Your Gemini API key.
- `DATASET_PATH`: Path to the input CSV file.

## Output

- The script prints a progress bar and a final comparison table to the console.
- **`report.md`**: A markdown report with the comparison table and discussion.
- **`evaluation_results_detailed.csv`**: Detailed CSV with predictions for each review.
- **`evaluation_metrics.csv`**: CSV containing the summary metrics.
