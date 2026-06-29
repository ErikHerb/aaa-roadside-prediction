# AAA Member Service Prediction & Travel Segmentation

Predicting roadside-assistance usage and segmenting members for targeted outreach, using a PySpark machine-learning pipeline built on ~1.7 million member records.

## Overview

AAA wanted to understand which members are likely to use roadside assistance and how to focus its travel marketing more efficiently. This project builds a member-level dataset from multiple source tables, explores the patterns behind service usage, trains classification models to predict roadside-assistance usage, and adds a clustering analysis to group members by travel behavior.

**Business question:** *Can we predict whether an active AAA member has used roadside assistance, and can we segment members to help the Travel Department prioritize outreach?*

## Dataset

- **Population:** 1,687,045 active AAA members, aggregated to one row per member
- **Target:** roadside-assistance usage (used = 721,588 / ~42.8%, did not use = 965,457)
- **Feature groups:**
  - *Membership profile* — tenure, category, coverage level, renewal method, region
  - *Demographics* — age group, income band, gender, education, marital status, home ownership
  - *Service behavior* — branch, discounts, insurance, travel, Visa, mortgage, tire/wheel activity
  - *Aggregated activity* — usage flags, counts, totals, averages, recency measures

> **Note:** The underlying member data is proprietary and is **not** included in this repository. The notebook is provided for methodology and code review.

## Approach

1. **Data cleaning & prep** — aggregate to one row per member, select active members, standardize strings, handle missing values, drop duplicates, and cast date types.
2. **Exploratory analysis** — service-adoption rates, roadside usage by income band, and roadside usage by member tenure to motivate the modeling direction.
3. **Classification** — a Spark ML pipeline using `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`, with an 80/20 train/test split, comparing **Logistic Regression**, **Random Forest**, and a **tuned Random Forest** (`ParamGridBuilder` + `TrainValidationSplit`).
4. **Evaluation** — AUC, accuracy, precision, recall, F1, and confusion-matrix counts, plus a cumulative-gains curve and a coefficient-direction view for interpretation.
5. **Segmentation** — a **K-Means** clustering on travel-related behavior to identify member segments.

## Key Results

- **Logistic Regression** produced the strongest AUC in this run; tuning improved the Random Forest over its untuned version.
- The gains curve showed the model ranks members meaningfully better than random, so outreach can be focused on higher-probability members.
- **Member tenure** was the single strongest predictor, followed by demographic signals (age, income, home ownership, marital status).
- K-Means surfaced three travel segments — *High-Value Travel Opportunity*, *Selective Travel*, and *Low Travel Engagement* — giving the Travel Department a practical targeting framework.
- **Recall** was the main limitation and the clearest area for future improvement.

## Results in Detail

The aggregate results and the script that produces these charts are in
[`results/`](results/) — no member-level data is included.

**Model comparison (roadside-assistance classification):**

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression* | ≈0.63 | ≈0.60 | ≈0.57 | ≈0.31 | ≈0.40 |
| Random Forest | 0.628 | 0.588 | 0.623 | 0.097 | 0.168 |
| Tuned Random Forest | 0.634 | 0.597 | 0.603 | 0.174 | 0.270 |

\* Logistic Regression figures are approximate (read from the presentation chart); the others are exact. Logistic Regression led on AUC, recall, and F1, making it the most useful model for outreach.

![Model performance comparison](results/model_performance_comparison.png)

**Most important predictors** — member tenure dominated, followed by demographic signals:

![Feature importance](results/feature_importance.png)

**Travel segments (K-Means):** clustering split members into three actionable groups for the Travel Department.

| Segment | Members | Travel usage rate |
|---|---|---|
| High-Value Travel Opportunity | 441,974 | 9.02% |
| Selective Travel | 25,728 | 8.49% |
| Low Travel Engagement | 1,219,343 | 2.99% |

![Travel segments](results/travel_segments.png)

## Tech Stack

Python · PySpark (Spark MLlib, Spark SQL) · pandas · matplotlib

## Repository Contents

- `Project_Update_Final_Notebook.ipynb` — full pipeline: cleaning, EDA, classification, and clustering
- `helper_functions.py` — shared helper utilities used by the notebook *(add this file)*

## How to Run

This notebook requires a Spark environment (e.g., Databricks or a local PySpark install) and access to the member tables.

```bash
pip install pyspark pandas matplotlib
```

Open the notebook in a Spark-enabled environment, point the load step at your data tables, and run the cells in order.

## Team & Role

Group project for **MSDS 630 — Large-Scale Data Analytics**. Team: Grant Robinson, Erik Herb, Jackson Swallow, Simon Salaj.

## Limitations & Next Steps

- Uses only active members, so it does not capture members who have left.
- Predicts *past* usage rather than future need; a time-bounded target would be a stronger framing.
- Tenure partly reflects time enrolled as well as true behavior.
- Next steps: include former members, add gradient boosting, and reframe the target around future behavior.
