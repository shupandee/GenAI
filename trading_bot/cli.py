#!/usr/bin/env python3
"""CLI entry point for the trading bot."""
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bot import BinanceClient, OrderManager, validate_order_input, setup_logging


app = typer.Typer(
    name="trading-bot",
    help="Binance Futures Testnet Trading Bot",
    add_completion=False
)
console = Console()


def get_credentials():
    """Get API credentials from environment variables."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        console.print("[red]Error:[/red] API credentials not found!")
        console.print("Please set the following environment variables:")
        console.print("  - BINANCE_API_KEY")
        console.print("  - BINANCE_API_SECRET")
        console.print("\nExample:")
        console.print("  export BINANCE_API_KEY='your_api_key'")
        console.print("  export BINANCE_API_SECRET='your_api_secret'")
        raise typer.Exit(code=1)
    
    return api_key, api_secret


@app.command()
def place_order(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDT)"),
    side: str = typer.Argument(..., help="Order side: BUY or SELL"),
    order_type: str = typer.Argument(..., help="Order type: MARKET or LIMIT"),
    quantity: float = typer.Argument(..., help="Order quantity"),
    price: Optional[float] = typer.Option(None, "--price", "-p", help="Limit price (required for LIMIT orders)"),
):
    """
    Place a MARKET or LIMIT order on Binance Futures Testnet.
    
    Examples:
    
    Place a MARKET order:
        trading-bot place-order BTCUSDT BUY MARKET 0.001
    
    Place a LIMIT order:
        trading-bot place-order BTCUSDT SELL LIMIT 0.001 --price 50000
    """
    # Setup logging
    logger = setup_logging()
    
    try:
        # Display header
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Binance Futures Testnet Trading Bot[/bold cyan]",
            border_style="cyan"
        ))
        console.print()
        
        # Validate input
        console.print("[yellow]Validating input...[/yellow]")
        order_input = validate_order_input(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        console.print("[green]✓[/green] Input validated successfully")
        console.print()
        
        # Display order summary
        table = Table(title="Order Summary", show_header=False, border_style="blue")
        table.add_column("Field", style="cyan", width=15)
        table.add_column("Value", style="white")
        
        table.add_row("Symbol", order_input.symbol)
        table.add_row("Side", f"[green]{order_input.side}[/green]" if order_input.side == "BUY" else f"[red]{order_input.side}[/red]")
        table.add_row("Order Type", order_input.order_type)
        table.add_row("Quantity", str(order_input.quantity))
        if order_input.price:
            table.add_row("Price", str(order_input.price))
        
        console.print(table)
        console.print()
        
        # Get credentials
        console.print("[yellow]Loading API credentials...[/yellow]")
        api_key, api_secret = get_credentials()
        console.print("[green]✓[/green] Credentials loaded")
        console.print()
        
        # Initialize client
        console.print("[yellow]Connecting to Binance Futures Testnet...[/yellow]")
        client = BinanceClient(api_key, api_secret)
        
        # Test connection
        try:
            client.ping()
            console.print("[green]✓[/green] Connected successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] Connection failed: {e}")
            raise
        
        console.print()
        
        # Place order
        order_manager = OrderManager(client)
        console.print(f"[yellow]Placing {order_input.order_type} order...[/yellow]")
        
        response = order_manager.place_order(order_input)
        
        # Display result
        console.print()
        result_table = Table(title="Order Result", show_header=False, border_style="green")
        result_table.add_column("Field", style="cyan", width=20)
        result_table.add_column("Value", style="white")
        
        result_table.add_row("Order ID", str(response.get('orderId', 'N/A')))
        result_table.add_row("Client Order ID", str(response.get('clientOrderId', 'N/A')))
        result_table.add_row("Status", f"[green]{response.get('status', 'N/A')}[/green]")
        result_table.add_row("Symbol", response.get('symbol', 'N/A'))
        result_table.add_row("Side", response.get('side', 'N/A'))
        result_table.add_row("Type", response.get('type', 'N/A'))
        result_table.add_row("Original Quantity", str(response.get('origQty', 'N/A')))
        result_table.add_row("Executed Quantity", str(response.get('executedQty', 'N/A')))
        
        if response.get('avgPrice'):
            result_table.add_row("Average Price", str(response.get('avgPrice')))
        if response.get('price'):
            result_table.add_row("Limit Price", str(response.get('price')))
        
        console.print(result_table)
        console.print()
        
        # Success message
        status = response.get('status', '')
        if status in ['NEW', 'FILLED', 'PARTIALLY_FILLED']:
            console.print(Panel(
                f"[bold green]✓ Order placed successfully![/bold green]\n"
                f"Order ID: {response.get('orderId')}",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]⚠ Order placed but status is: {status}[/bold yellow]\n"
                f"Order ID: {response.get('orderId')}",
                border_style="yellow"
            ))
        
        console.print()
        
    except ValueError as e:
        console.print()
        console.print(Panel(
            f"[bold red]Validation Error[/bold red]\n{str(e)}",
            border_style="red"
        ))
        logger.error(f"Validation error: {e}")
        raise typer.Exit(code=1)
    
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Error[/bold red]\n{str(e)}",
            border_style="red"
        ))
        logger.error(f"Order placement failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def test_connection():
    """Test connection to Binance Futures Testnet API."""
    logger = setup_logging()
    
    try:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Testing Binance Futures Testnet Connection[/bold cyan]",
            border_style="cyan"
        ))
        console.print()
        
        # Get credentials
        console.print("[yellow]Loading API credentials...[/yellow]")
        api_key, api_secret = get_credentials()
        console.print("[green]✓[/green] Credentials loaded")
        console.print()
        
        # Initialize client
        client = BinanceClient(api_key, api_secret)
        
        # Test ping
        console.print("[yellow]Testing API connectivity (ping)...[/yellow]")
        client.ping()
        console.print("[green]✓[/green] Ping successful")
        console.print()
        
        # Test server time
        console.print("[yellow]Fetching server time...[/yellow]")
        time_response = client.get_server_time()
        console.print(f"[green]✓[/green] Server time: {time_response.get('serverTime')}")
        console.print()
        
        # Test account info
        console.print("[yellow]Fetching account information...[/yellow]")
        account = client.get_account_info()
        console.print(f"[green]✓[/green] Account retrieved successfully")
        console.print(f"  Assets: {len(account.get('assets', []))}")
        console.print(f"  Positions: {len(account.get('positions', []))}")
        console.print()
        
        console.print(Panel(
            "[bold green]✓ All connection tests passed![/bold green]",
            border_style="green"
        ))
        console.print()
        
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Connection Test Failed[/bold red]\n{str(e)}",
            border_style="red"
        ))
        logger.error(f"Connection test failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def version():
    """Display version information."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Binance Futures Testnet Trading Bot[/bold cyan]\n"
        "Version: 1.0.0\n"
        "Author: Trading Bot Team",
        border_style="cyan"
    ))
    console.print()


if __name__ == "__main__":
    app()