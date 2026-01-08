import os
from dotenv import load_dotenv

from binance.client import Client
from binance.exceptions import BinanceAPIException


def main() -> None:
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    if not api_key or not api_secret:
        print("Missing BINANCE_API_KEY or BINANCE_SECRET in environment.")
        return

    client = Client(
        api_key=api_key,
        api_secret=api_secret,
        requests_params={"timeout": 10},
        verbose=True,
    )

    try:
        print("[api_check] Fetching account balances...")
        account = client.get_account()
        usdt_balance = next(
            (balance.get("free") for balance in account.get("balances", []) if balance.get("asset") == "USDT"),
            "0",
        )
        print(f"API check OK. Available USDT balance: {usdt_balance}")
    except BinanceAPIException as exc:
        print("API check failed with BinanceAPIException:", exc)
    except Exception as exc:
        print("API check failed with unexpected error:", exc)


if __name__ == "__main__":
    main()
