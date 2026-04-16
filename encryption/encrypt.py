# Apex Iris Framework - Encryption Module
import sys
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

def encrypt_file(input_file_path):
    """
    Encrypts a single file using hybrid AES (Fernet) + RSA encryption.
    The encrypted file is saved as <filename>.enc
    """
    _dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(_dir, "public_key.pem")

    # Load RSA public key
    with open(key_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())

    # Generate a random session key (Fernet)
    session_key = Fernet.generate_key()
    cipher_suite = Fernet(session_key)

    # Read input file
    with open(input_file_path, "rb") as f:
        file_data = f.read()

    # Encrypt file data using Fernet
    encrypted_data = cipher_suite.encrypt(file_data)

    # Encrypt session key using RSA
    encrypted_session_key = public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Write encrypted session key + encrypted data to output file
    output_file = input_file_path + ".enc"
    with open(output_file, "wb") as f:
        f.write(encrypted_session_key)
        f.write(encrypted_data)

    print(f"Success! Encrypted '{input_file_path}' → '{output_file}' (Hybrid Mode).")


# Optional: allow running a single file from CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python encrypt.py <filename>")
    else:
        encrypt_file(sys.argv[1])
