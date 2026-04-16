# scripts/leaderboard/render_leaderboard.py
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "leaderboard.csv"
MD_PATH = BASE_DIR / "leaderboard.md"
DOCS_CSV_PATH = BASE_DIR.parent / "docs" / "leaderboard.csv"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing leaderboard CSV at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    DOCS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOCS_CSV_PATH, index=False)

    if df.empty:
        MD_PATH.write_text("# Leaderboard\n\n_No submissions yet._\n", encoding="utf-8")
        return

    # Sort same as leaderboard
    df = df.sort_values(
        ["validation_f1_perturbed", "robustness_gap"],
        ascending=[False, True]
    )
    df.insert(0, "rank", range(1, len(df) + 1))

    md_content = ["# Leaderboard", "", df.to_markdown(index=False)]
    MD_PATH.write_text("\n".join(md_content) + "\n", encoding="utf-8")
    print(f"Rendered {MD_PATH}")

if __name__ == "__main__":
    main()
