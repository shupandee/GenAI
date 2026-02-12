#!/usr/bin/env python3
"""
Example usage script for the trading bot.

This script demonstrates various usage patterns.
Make sure to set your API credentials before running:
    export BINANCE_API_KEY='your_key'
    export BINANCE_API_SECRET='your_secret'
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import BinanceClient, OrderManager, validate_order_input, setup_logging


def example_market_order():
    """Example: Place a MARKET order."""
    print("\n" + "="*80)
    print("EXAMPLE 1: MARKET Order")
    print("="*80 + "\n")
    
    # Setup
    logger = setup_logging()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    # Initialize
    client = BinanceClient(api_key, api_secret)
    order_manager = OrderManager(client)
    
    # Validate input
    order_input = validate_order_input(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.001
    )
    
    # Place order
    response = order_manager.place_order(order_input)
    
    print(f"\nOrder placed! ID: {response['orderId']}")
    print(f"Status: {response['status']}")
    print(f"Executed: {response['executedQty']} @ {response.get('avgPrice', 'N/A')}")


def example_limit_order():
    """Example: Place a LIMIT order."""
    print("\n" + "="*80)
    print("EXAMPLE 2: LIMIT Order")
    print("="*80 + "\n")
    
    # Setup
    logger = setup_logging()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    # Initialize
    client = BinanceClient(api_key, api_secret)
    order_manager = OrderManager(client)
    
    # Validate input
    order_input = validate_order_input(
        symbol="ETHUSDT",
        side="SELL",
        order_type="LIMIT",
        quantity=0.01,
        price=3200.50
    )
    
    # Place order
    response = order_manager.place_order(order_input)
    
    print(f"\nOrder placed! ID: {response['orderId']}")
    print(f"Status: {response['status']}")
    print(f"Limit Price: {response['price']}")


def example_validation_error():
    """Example: Handling validation errors."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Validation Error Handling")
    print("="*80 + "\n")
    
    logger = setup_logging()
    
    try:
        # This will fail - missing price for LIMIT order
        order_input = validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001
            # price is missing!
        )
    except ValueError as e:
        print(f"Validation error caught: {e}")
        print("This is expected - LIMIT orders require a price parameter")


def example_get_account_info():
    """Example: Get account information."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Get Account Information")
    print("="*80 + "\n")
    
    logger = setup_logging()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    # Initialize
    client = BinanceClient(api_key, api_secret)
    
    # Get account info
    account = client.get_account_info()
    
    print(f"Total Wallet Balance: {account.get('totalWalletBalance')} USDT")
    print(f"Available Balance: {account.get('availableBalance')} USDT")
    print(f"Number of positions: {len([p for p in account.get('positions', []) if float(p.get('positionAmt', 0)) != 0])}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BINANCE FUTURES TESTNET TRADING BOT - EXAMPLES")
    print("="*80)
    
    # Check credentials
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        print("\nERROR: API credentials not found!")
        print("Please set environment variables:")
        print("  export BINANCE_API_KEY='your_key'")
        print("  export BINANCE_API_SECRET='your_secret'")
        sys.exit(1)
    
    print("\nRunning examples...\n")
    
    # Run examples (commented out to avoid accidental execution)
    # Uncomment the ones you want to run:
    
    # example_market_order()
    # example_limit_order()
    example_validation_error()
    example_get_account_info()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("Check the logs/ directory for detailed logs")
    print("="*80 + "\n")