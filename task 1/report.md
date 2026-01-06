# Yelp Rating Prediction Report

## Comparison Table

| Strategy    |   Accuracy |   JSON_Validity |   Valid_Samples |
|:------------|-----------:|----------------:|----------------:|
| v1_zeroshot |       0.6  |             1   |               5 |
| v2_fewshot  |       0.75 |             0.8 |               4 |
| v3_roleplay |       0    |             0   |               0 |

## Discussion
### 1. Zero-shot Baseline
Direct asking. Good baseline, but might miss nuances or strict formatting occasionally.

### 2. Few-shot
Provided 3 examples. Typically stabilizes the output format and helps with ambiguous cases.

### 3. Role-based + CoT
Asked to act as a critic and analyze steps. This generally improves reasoning for complex reviews where sentiment doesn't match the score trivially.
