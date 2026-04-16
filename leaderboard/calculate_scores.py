# Updated by Satyam
# leaderboard/calculate_scores.py
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
import os
import sys  # add this import

# Get test labels from environment variable (set in workflow)
TEST_LABELS_PATH = os.environ.get('TEST_LABELS_CSV')

def calculate_scores(submission_path: Path):
    """
    Compute F1 score for a single CSV submission and return as dict
    """
    print(f"DEBUG: calculate_scores called with submission: {submission_path}", file=sys.stderr)
    
    # Check if file exists
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    
    print(f"DEBUG: Loading submission from {submission_path}", file=sys.stderr)
    submission_df = pd.read_csv(submission_path)
    
    print(f"DEBUG: Submission columns: {list(submission_df.columns)}", file=sys.stderr)
    print(f"DEBUG: Submission shape: {submission_df.shape}", file=sys.stderr)
    print(f"DEBUG: First few rows of submission:", file=sys.stderr)
    print(submission_df.head(3).to_string(), file=sys.stderr)
    
    # Check for graph_index column
    if "graph_index" not in submission_df.columns:
        raise ValueError(f"Submission missing required column: graph_index. Found columns: {list(submission_df.columns)}")
    
    # Find the prediction column
    possible_pred_cols = ["label", "prediction", "target", "predictions", "Label", "Prediction", "Target", "y_pred", "pred"]
    
    pred_col = None
    for col in possible_pred_cols:
        if col in submission_df.columns:
            pred_col = col
            print(f"DEBUG: Found prediction column: '{col}'", file=sys.stderr)
            break
    
    if pred_col is None:
        other_cols = [col for col in submission_df.columns if col != "graph_index"]
        if len(other_cols) == 1:
            pred_col = other_cols[0]
            print(f"DEBUG: Using '{pred_col}' as prediction column", file=sys.stderr)
        else:
            raise ValueError(f"Could not find prediction column. Found: {list(submission_df.columns)}")
    
    # Load test labels from environment variable
    print(f"DEBUG: TEST_LABELS_CSV = {TEST_LABELS_PATH}", file=sys.stderr)
    if not TEST_LABELS_PATH:
        raise ValueError("TEST_LABELS_CSV environment variable not set!")
    
    test_labels_path = Path(TEST_LABELS_PATH)
    if not test_labels_path.exists():
        raise FileNotFoundError(f"Test labels file not found: {test_labels_path}")
    
    print(f"DEBUG: Loading test labels from {test_labels_path}", file=sys.stderr)
    gt_df = pd.read_csv(test_labels_path)
    print(f"DEBUG: Test labels columns: {list(gt_df.columns)}", file=sys.stderr)
    print(f"DEBUG: Test labels shape: {gt_df.shape}", file=sys.stderr)
    print(f"DEBUG: First few rows of test labels:", file=sys.stderr)
    print(gt_df.head(3).to_string(), file=sys.stderr)
    
    # Find ground truth label column
    possible_truth_cols = ["label", "target", "Label", "Target"]
    truth_col = None
    for col in possible_truth_cols:
        if col in gt_df.columns:
            truth_col = col
            print(f"DEBUG: Found ground truth column: '{col}'", file=sys.stderr)
            break
    
    if truth_col is None:
        other_cols = [col for col in gt_df.columns if col != "graph_index"]
        if len(other_cols) == 1:
            truth_col = other_cols[0]
            print(f"DEBUG: Using '{truth_col}' as ground truth column", file=sys.stderr)
        else:
            raise ValueError(f"Could not find ground truth column. Found: {list(gt_df.columns)}")
    
    # Merge on graph_index
    print(f"DEBUG: Merging on graph_index...", file=sys.stderr)
    merged = submission_df.merge(gt_df, on="graph_index", how="inner")
    print(f"DEBUG: Merged shape: {merged.shape}", file=sys.stderr)
    
    if len(merged) == 0:
        print(f"DEBUG: Submission graph_index sample: {submission_df['graph_index'].head()}", file=sys.stderr)
        print(f"DEBUG: Test labels graph_index sample: {gt_df['graph_index'].head()}", file=sys.stderr)
        raise ValueError("No matching graph_index values found between submission and test labels")
    
    y_pred = merged[pred_col]
    y_true = merged[truth_col]
    
    print(f"DEBUG: y_pred sample: {y_pred.head().tolist()}", file=sys.stderr)
    print(f"DEBUG: y_true sample: {y_true.head().tolist()}", file=sys.stderr)
    
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"DEBUG: Calculated F1 score: {f1}", file=sys.stderr)
    
    return {"validation_f1_score": f1}
