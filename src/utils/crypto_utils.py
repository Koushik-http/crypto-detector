import hashlib
import requests

class CryptoUtils:
    @staticmethod
    def validate_crypto_address(address: str, prefix: str = '1') -> bool:
        """
        Validates a cryptocurrency address based on the prefix.
        :param address: Cryptocurrency address to validate.
        :param prefix: Expected prefix of the address (e.g., '1' for Bitcoin).
        :return: True if valid, False otherwise.
        """
        if not isinstance(address, str):
            raise TypeError("Address must be a string.")
        return address.startswith(prefix)

    @staticmethod
    def hash_transaction_data(transaction_data: dict) -> str:
        """
        Creates a cryptographic hash (SHA-256) of the transaction data.
        :param transaction_data: A dictionary containing transaction details.
        :return: SHA-256 hash of the transaction data.
        """
        if not isinstance(transaction_data, dict):
            raise TypeError("Transaction data must be a dictionary.")
        transaction_string = str(sorted(transaction_data.items()))
        return hashlib.sha256(transaction_string.encode()).hexdigest()

    @staticmethod
    def convert_crypto_to_fiat(crypto_amount: float, crypto_symbol: str, fiat_symbol: str = 'USD') -> float:
        """
        Converts a cryptocurrency amount to fiat using a live exchange rate API.
        :param crypto_amount: The amount of cryptocurrency to convert.
        :param crypto_symbol: Symbol of the cryptocurrency (e.g., BTC, ETH).
        :param fiat_symbol: Symbol of the fiat currency (e.g., USD, EUR).
        :return: Equivalent fiat amount.
        """
        if not isinstance(crypto_amount, (int, float)) or crypto_amount < 0:
            raise ValueError("Crypto amount must be a non-negative number.")
        if not isinstance(crypto_symbol, str) or not isinstance(fiat_symbol, str):
            raise TypeError("Crypto symbol and fiat symbol must be strings.")
        
        try:
            response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_symbol}&vs_currencies={fiat_symbol}", timeout=10)
            response.raise_for_status()  # Raise an error for HTTP errors
            data = response.json()
            if crypto_symbol.lower() not in data:
                raise ValueError(f"Cryptocurrency '{crypto_symbol}' not found in response.")
            rate = data[crypto_symbol.lower()][fiat_symbol.lower()]
            return crypto_amount * rate
        except requests.RequestException as e:
            raise ConnectionError(f"Error fetching conversion rate: {e}")

    @staticmethod
    def calculate_transaction_fee(transaction_size_bytes: int, fee_rate_per_byte: float) -> float:
        """
        Calculates the transaction fee based on the transaction size and fee rate per byte.
        :param transaction_size_bytes: The size of the transaction in bytes.
        :param fee_rate_per_byte: The fee rate per byte (e.g., in satoshis for Bitcoin).
        :return: The total transaction fee.
        """
        if not isinstance(transaction_size_bytes, int) or transaction_size_bytes < 0:
            raise ValueError("Transaction size must be a non-negative integer.")
        if not isinstance(fee_rate_per_byte, (int, float)) or fee_rate_per_byte < 0:
            raise ValueError("Fee rate must be a non-negative number.")
        return transaction_size_bytes * fee_rate_per_byte

    @staticmethod
    def generate_crypto_address(public_key: str) -> str:
        """
        Generates a cryptocurrency address from a public key using hashing techniques.
        :param public_key: The public key of the user.
        :return: A cryptocurrency address.
        """
        if not isinstance(public_key, str):
            raise TypeError("Public key must be a string.")
        
        sha256_hash = hashlib.sha256(public_key.encode()).hexdigest()
        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(sha256_hash.encode())
        return ripemd160.hexdigest()

    @staticmethod
    def fetch_current_crypto_price(crypto_symbol: str, fiat_symbol: str = 'USD') -> float:
        """
        Fetches the current price of a cryptocurrency in fiat using a live exchange rate API.
        :param crypto_symbol: The symbol of the cryptocurrency (e.g., BTC, ETH).
        :param fiat_symbol: The symbol of the fiat currency (e.g., USD).
        :return: The current price of the cryptocurrency in the specified fiat currency.
        """
        if not isinstance(crypto_symbol, str) or not isinstance(fiat_symbol, str):
            raise TypeError("Crypto symbol and fiat symbol must be strings.")
        
        try:
            response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_symbol}&vs_currencies={fiat_symbol}", timeout=10)
            response.raise_for_status()  # Raise an error for HTTP errors
            data = response.json()
            if crypto_symbol.lower() not in data:
                raise ValueError(f"Cryptocurrency '{crypto_symbol}' not found in response.")
            return data[crypto_symbol.lower()][fiat_symbol.lower()]
        except requests.RequestException as e:
            raise ConnectionError(f"Error fetching price: {e}")
