import os
import sys

# Add parent folder to Python path to find encryption package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encryption.encrypt import encrypt_file

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))

# Get all CSV files, skip sample_submission.csv, skip already encrypted files
csv_files = [
    f for f in os.listdir(SUBMISSION_DIR)
    if f.endswith(".csv") and f != "sample_submission.csv"
    and not os.path.exists(os.path.join(SUBMISSION_DIR, f + ".enc"))
]

# Encrypt each CSV file
for csv_file in csv_files:
    input_path = os.path.join(SUBMISSION_DIR, csv_file)
    encrypt_file(input_path)
