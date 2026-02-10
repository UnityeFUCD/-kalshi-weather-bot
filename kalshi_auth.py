"""
kalshi_auth.py — RSA-PSS request signing for Kalshi API.

Implements the exact auth protocol from:
https://docs.kalshi.com/getting_started/api_keys

Each request is independently signed. No sessions/tokens needed.

Signing message format: {timestamp_ms}{METHOD}{path_without_query_params}
"""

import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


def load_private_key(file_path: str):
    """Load RSA private key from PEM file."""
    with open(file_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )


def sign_request(private_key, method: str, path: str) -> dict:
    """
    Generate signed headers for a Kalshi API request.
    
    Args:
        private_key: Loaded RSA private key object
        method: HTTP method (GET, POST, DELETE)
        path: Request path WITHOUT query params (e.g., /trade-api/v2/markets)
    
    Returns:
        Dict of headers to include in request:
        {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms_str,
            "KALSHI-ACCESS-SIGNATURE": base64_signature
        }
    """
    # Current time in milliseconds
    timestamp_ms = str(int(time.time() * 1000))
    
    # CRITICAL: Strip query params before signing
    # From docs: "use the path without query parameters"
    path_clean = path.split("?")[0]
    
    # Build message: timestamp + METHOD + path
    message = f"{timestamp_ms}{method.upper()}{path_clean}"
    
    # Sign with RSA-PSS (SHA256, salt_length = digest_length)
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    
    return {
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
    }


class KalshiAuth:
    """
    Reusable auth object. Loads key once, signs every request.
    
    Usage:
        auth = KalshiAuth("key-id", "path/to/private_key.pem")
        headers = auth.headers("GET", "/trade-api/v2/portfolio/balance")
        response = requests.get(url, headers=headers)
    """
    
    def __init__(self, api_key_id: str, private_key_path: str):
        self.api_key_id = api_key_id
        self.private_key = load_private_key(private_key_path)
    
    def headers(self, method: str, path: str) -> dict:
        """Generate complete auth headers for a request."""
        h = sign_request(self.private_key, method, path)
        h["KALSHI-ACCESS-KEY"] = self.api_key_id
        h["Content-Type"] = "application/json"
        return h


if __name__ == "__main__":
    # Quick test: load key and generate a signature
    import config
    
    auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
    test_headers = auth.headers("GET", "/trade-api/v2/portfolio/balance")
    
    print("Auth test successful!")
    print(f"  Key ID: {auth.api_key_id[:8]}...")
    print(f"  Timestamp: {test_headers['KALSHI-ACCESS-TIMESTAMP']}")
    print(f"  Signature: {test_headers['KALSHI-ACCESS-SIGNATURE'][:40]}...")
