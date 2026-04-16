# leaderboard/score_submission.py
import argparse
import json
from pathlib import Path
import sys
import os   # add this import if you want the debug line below

# Add repo root to sys.path
repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from leaderboard.calculate_scores import calculate_scores

def validate_metadata(submission_path: Path):
    metadata_path = submission_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json next to {submission_path.name}")
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json is invalid JSON: {exc}") from exc

def main():
    parser = argparse.ArgumentParser(description="Score a single submission file.")
    parser.add_argument("submission_path", help="Path to predictions CSV")
    parser.add_argument("--require-metadata", action="store_true", help="Require metadata.json")
    args = parser.parse_args()

    submission_path = Path(args.submission_path).resolve()
    if args.require_metadata:
        validate_metadata(submission_path)

    # Optional debug (prints to stderr, not affecting JSON)
    print(f"DEBUG: TEST_LABELS_CSV = {os.environ.get('TEST_LABELS_CSV')}", file=sys.stderr)

    scores = calculate_scores(submission_path)
    # Only JSON to stdout
    print(json.dumps(scores))

if __name__ == "__main__":
    main()
