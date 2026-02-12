# Binance Futures Testnet Trading Bot

A professional Python trading bot for placing orders on Binance Futures Testnet (USDT-M). Features clean architecture, comprehensive logging, input validation, and an enhanced CLI experience.

## Features

- ✅ Place MARKET and LIMIT orders on Binance Futures Testnet
- ✅ Support for BUY and SELL sides
- ✅ Comprehensive input validation with Pydantic
- ✅ Structured logging to files with detailed request/response tracking
- ✅ Clean separation of concerns (client, orders, validators, CLI)
- ✅ Enhanced CLI with Rich library for beautiful output
- ✅ Robust error handling for API errors, network failures, and invalid input
- ✅ Type hints throughout the codebase
- ✅ Professional code structure following Python best practices

## Project Structure

```
trading_bot/
├── bot/
│   ├── __init__.py          # Package initialization
│   ├── client.py            # Binance API client wrapper
│   ├── orders.py            # Order placement logic
│   ├── validators.py        # Input validation with Pydantic
│   └── logging_config.py    # Logging configuration
├── cli.py                   # CLI entry point (Typer)
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment variables
├── README.md                # This file
└── logs/                    # Log files (created automatically)
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Binance Futures Testnet account

### 2. Create Binance Futures Testnet Account

1. Visit https://testnet.binancefuture.com
2. Create an account
3. Generate API credentials from the account dashboard
4. Save your API Key and API Secret

### 3. Clone or Download

```bash
# If using git
git clone <repository-url>
cd trading_bot

# Or extract from zip
unzip trading_bot.zip
cd trading_bot
```

### 4. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure API Credentials

**Option 1: Environment Variables (Recommended)**

```bash
# Linux/Mac
export BINANCE_API_KEY='your_api_key_here'
export BINANCE_API_SECRET='your_api_secret_here'

# Windows Command Prompt
set BINANCE_API_KEY=your_api_key_here
set BINANCE_API_SECRET=your_api_secret_here

# Windows PowerShell
$env:BINANCE_API_KEY='your_api_key_here'
$env:BINANCE_API_SECRET='your_api_secret_here'
```

**Option 2: .env File**

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your credentials
nano .env  # or use any text editor
```

Then load it before running:
```bash
source .env  # Linux/Mac
```

## Usage

### Basic Commands

The bot provides three main commands:

1. **place-order** - Place a MARKET or LIMIT order
2. **test-connection** - Test API connectivity
3. **version** - Display version information

### Test Connection

Before placing orders, verify your API credentials work:

```bash
python cli.py test-connection
```

### Place MARKET Order

#### Example 1: Buy BTC with MARKET order

```bash
python cli.py place-order BTCUSDT BUY MARKET 0.001
```

This will:
- Buy 0.001 BTC
- Execute at current market price
- Complete immediately

#### Example 2: Sell ETH with MARKET order

```bash
python cli.py place-order ETHUSDT SELL MARKET 0.01
```

### Place LIMIT Order

#### Example 3: Buy BTC with LIMIT order at specific price

```bash
python cli.py place-order BTCUSDT BUY LIMIT 0.001 --price 45000
```

This will:
- Place a limit buy order for 0.001 BTC at $45,000
- Order remains open until price reaches $45,000 or you cancel it

#### Example 4: Sell ETH with LIMIT order

```bash
python cli.py place-order ETHUSDT SELL LIMIT 0.01 --price 3000
```

### View Help

```bash
# General help
python cli.py --help

# Command-specific help
python cli.py place-order --help
```

## Understanding the Output

### Order Request Summary
Shows what you're about to execute:
- Symbol
- Side (BUY/SELL)
- Order Type (MARKET/LIMIT)
- Quantity
- Price (for LIMIT orders)

### Order Response
Shows the result from Binance:
- Order ID (unique identifier)
- Status (NEW, FILLED, PARTIALLY_FILLED)
- Executed Quantity
- Average Price (for MARKET orders)
- Timestamp

### Success/Failure Messages
- ✅ Green = Order placed successfully
- ⚠️ Yellow = Order placed but unusual status
- ❌ Red = Order failed with error details

## Logging

All operations are logged to timestamped files in the `logs/` directory:

```
logs/
├── trading_bot_20260212_143022.log
├── trading_bot_20260212_143145.log
└── ...
```

Each log file contains:
- Timestamp for every operation
- API requests and responses
- Order details
- Error messages with stack traces
- Validation information

**Log Levels:**
- DEBUG: Detailed API request/response data
- INFO: General operation flow and status
- WARNING: Unusual but non-critical issues
- ERROR: Failures and exceptions

## Error Handling

The bot handles various error scenarios:

### 1. Validation Errors
```
Validation Error
Price is required for LIMIT orders
```
**Solution:** Add `--price` flag for LIMIT orders

### 2. API Errors
```
HTTP Error: 400 Client Error
Invalid quantity
```
**Solution:** Check symbol's minimum quantity requirements

### 3. Authentication Errors
```
Error: API credentials not found!
```
**Solution:** Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables

### 4. Network Errors
```
Request failed: Connection timeout
```
**Solution:** Check internet connection and try again

## Assumptions

1. **Testnet Only**: This bot is designed exclusively for Binance Futures Testnet and uses the testnet base URL (https://testnet.binancefuture.com)

2. **USDT-M Futures**: The bot works with USDT-margined futures contracts (symbols ending in USDT like BTCUSDT, ETHUSDT)

3. **Time in Force**: LIMIT orders use "GTC" (Good Till Cancel) by default

4. **API Credentials**: Assumed to be stored in environment variables for security

5. **Quantity Precision**: Users must provide valid quantities according to Binance's symbol requirements (the bot validates but doesn't auto-adjust)

6. **Python Version**: Requires Python 3.8+ for type hints and modern syntax

## Architecture Highlights

### Separation of Concerns
- **client.py**: Low-level API communication
- **orders.py**: Business logic for order management
- **validators.py**: Input validation and data models
- **cli.py**: User interface and command handling
- **logging_config.py**: Centralized logging setup

### Error Handling Strategy
1. Input validation before API calls
2. Try-except blocks around network operations
3. Detailed error logging with context
4. User-friendly error messages in CLI

### Security Practices
- API credentials from environment variables
- Secure HMAC SHA256 signature generation
- No hardcoded secrets
- Request signing for authenticated endpoints

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Permission denied on cli.py
```bash
chmod +x cli.py
```

### Invalid API credentials
1. Verify credentials on https://testnet.binancefuture.com
2. Ensure no extra spaces in environment variables
3. Check if API key has futures trading permissions

### Orders not executing
1. Check if symbol is valid: `python cli.py test-connection`
2. Verify you have testnet balance
3. Review log files for detailed error messages

## Next Steps / Enhancements

Potential improvements for production use:

1. **Additional Order Types**: Stop-Loss, Take-Profit, OCO
2. **Position Management**: View and close positions
3. **Risk Management**: Max position size, daily loss limits
4. **Configuration File**: YAML/JSON for trading parameters
5. **WebSocket Integration**: Real-time price feeds
6. **Backtesting**: Historical data testing
7. **Database**: Store order history
8. **Web Dashboard**: GUI for monitoring

## Support

For issues or questions:
1. Check the log files in `logs/` directory
2. Review Binance Futures Testnet API documentation
3. Verify your API credentials and permissions

## License

This project is provided as-is for educational and testing purposes on Binance Futures Testnet.

## Disclaimer

This bot is for **TESTNET USE ONLY**. Do not use with real funds without proper testing and risk management. Trading cryptocurrencies carries risk.