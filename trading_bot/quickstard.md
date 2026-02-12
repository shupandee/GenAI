# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- Binance Futures Testnet account ([Sign up here](https://testnet.binancefuture.com))

## Installation

```bash
# 1. Extract/Clone the project
cd trading_bot

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your API credentials as environment variables:

```bash
# Linux/Mac
export BINANCE_API_KEY='your_api_key_here'
export BINANCE_API_SECRET='your_api_secret_here'

# Windows (Command Prompt)
set BINANCE_API_KEY=your_api_key_here
set BINANCE_API_SECRET=your_api_secret_here

# Windows (PowerShell)
$env:BINANCE_API_KEY='your_api_key_here'
$env:BINANCE_API_SECRET='your_api_secret_here'
```

## Test Connection

```bash
python cli.py test-connection
```

You should see:
```
✓ Ping successful
✓ Server time: [timestamp]
✓ Account retrieved successfully
✓ All connection tests passed!
```

## Place Your First Order

### MARKET Order Example

```bash
python cli.py place-order BTCUSDT BUY MARKET 0.001
```

This will:
- Buy 0.001 BTC at current market price
- Execute immediately
- Show order confirmation

### LIMIT Order Example

```bash
python cli.py place-order ETHUSDT SELL LIMIT 0.01 --price 3200
```

This will:
- Place a sell order for 0.01 ETH at $3200
- Wait for price to reach $3200
- Show order ID and status

## Interactive Mode (Bonus Feature)

For a menu-driven experience:

```bash
python interactive.py
```

This provides:
- Visual menu interface
- Step-by-step prompts
- Order confirmation
- Account information viewer
- Open orders viewer

## View Help

```bash
# General help
python cli.py --help

# Command-specific help
python cli.py place-order --help
```

## Common Commands

```bash
# Test connection
python cli.py test-connection

# Market buy
python cli.py place-order BTCUSDT BUY MARKET 0.001

# Market sell
python cli.py place-order ETHUSDT SELL MARKET 0.01

# Limit buy
python cli.py place-order BTCUSDT BUY LIMIT 0.001 --price 45000

# Limit sell
python cli.py place-order ETHUSDT SELL LIMIT 0.01 --price 3200
```

## Check Logs

All operations are logged to the `logs/` directory:

```bash
# View latest log
ls -lt logs/ | head -2

# Read log file
cat logs/trading_bot_[timestamp].log
```

## Troubleshooting

### "API credentials not found"
Make sure you've set the environment variables correctly and they're available in your current terminal session.

### "Invalid quantity"
Check the minimum quantity requirements for your symbol on Binance Futures.

### "Module not found"
Run: `pip install -r requirements.txt`

## Next Steps

- Review the full [README.md](README.md) for detailed documentation
- Check [examples.py](examples.py) for programmatic usage
- Try the interactive mode for a better UX
- Review logs to understand API interactions

## Need Help?

1. Check the log files in `logs/` directory
2. Review the [README.md](README.md)
3. Verify your API credentials on https://testnet.binancefuture.com

## Important Notes

⚠️ **This is for TESTNET only** - Not for real trading
⚠️ **No real money involved** - All trades are simulated
⚠️ **Always review orders** before confirming

Happy testing! 🚀