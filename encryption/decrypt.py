# Apex Iris Framework - Encryption Module
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

load_dotenv()

def decrypt_file_content(encrypted_file_path: Path) -> bytes:
    private_key_pem = os.environ.get("SUBMISSION_PRIVATE_KEY")
    
    if not private_key_pem:
        raise ValueError("Error: 'SUBMISSION_PRIVATE_KEY' is missing from environment variables.")

    private_key_pem = private_key_pem.replace('\\n', '\n').strip()

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None
    )

    with open(encrypted_file_path, "rb") as f:
        file_content = f.read()

    rsa_segment_size = 256
    encrypted_session_key = file_content[:rsa_segment_size]
    encrypted_data = file_content[rsa_segment_size:]

    session_key = private_key.decrypt(
        encrypted_session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    cipher_suite = Fernet(session_key)
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    return decrypted_data

# Add this wrapper so update_leaderboard.py works
def decrypt_file(encrypted_file_path: Path, output_file_path: Path):
    decrypted_data = decrypt_file_content(encrypted_file_path)
    with open(output_file_path, "wb") as f:
        f.write(decrypted_data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decrypt.py <filename>")
        sys.exit(1)

    encrypted_file = Path(sys.argv[1])
    output_file = encrypted_file.with_suffix('')  # remove .enc

    try:
        decrypt_file(encrypted_file, output_file)
        print(f"Decryption successful! Saved to '{output_file}'")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
