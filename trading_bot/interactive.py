#!/usr/bin/env python3
"""Interactive menu mode for the trading bot (BONUS FEATURE)."""
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich.table import Table

from bot import BinanceClient, OrderManager, validate_order_input, setup_logging


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
        return None, None
    
    return api_key, api_secret


def display_header():
    """Display application header."""
    console.clear()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Binance Futures Testnet Trading Bot[/bold cyan]\n"
        "[dim]Interactive Mode[/dim]",
        border_style="cyan"
    ))
    console.print()


def display_menu():
    """Display main menu options."""
    table = Table(show_header=False, border_style="blue", box=None)
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Description", style="white")
    
    table.add_row("1", "Place MARKET Order")
    table.add_row("2", "Place LIMIT Order")
    table.add_row("3", "Test Connection")
    table.add_row("4", "View Account Info")
    table.add_row("5", "View Open Orders")
    table.add_row("0", "Exit")
    
    console.print(table)
    console.print()


def place_market_order(order_manager: OrderManager):
    """Interactive MARKET order placement."""
    console.print()
    console.print(Panel.fit("[bold]Place MARKET Order[/bold]", border_style="yellow"))
    console.print()
    
    try:
        # Get order details
        symbol = Prompt.ask("Enter symbol (e.g., BTCUSDT)", default="BTCUSDT").upper()
        
        side = Prompt.ask(
            "Enter side",
            choices=["BUY", "SELL", "buy", "sell"],
            default="BUY"
        ).upper()
        
        quantity = FloatPrompt.ask("Enter quantity")
        
        # Display order summary
        console.print()
        table = Table(title="Order Summary", show_header=False, border_style="blue")
        table.add_column("Field", style="cyan", width=15)
        table.add_column("Value", style="white")
        
        table.add_row("Symbol", symbol)
        table.add_row("Side", f"[green]{side}[/green]" if side == "BUY" else f"[red]{side}[/red]")
        table.add_row("Order Type", "MARKET")
        table.add_row("Quantity", str(quantity))
        
        console.print(table)
        console.print()
        
        # Confirm
        if not Confirm.ask("Proceed with this order?"):
            console.print("[yellow]Order cancelled[/yellow]")
            return
        
        # Validate and place order
        order_input = validate_order_input(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity
        )
        
        console.print()
        console.print("[yellow]Placing order...[/yellow]")
        response = order_manager.place_order(order_input)
        
        # Display result
        console.print()
        console.print(Panel(
            f"[bold green]✓ Order placed successfully![/bold green]\n"
            f"Order ID: {response.get('orderId')}\n"
            f"Status: {response.get('status')}\n"
            f"Executed: {response.get('executedQty')} @ {response.get('avgPrice', 'N/A')}",
            border_style="green"
        ))
        
    except ValueError as e:
        console.print()
        console.print(Panel(f"[bold red]Validation Error[/bold red]\n{str(e)}", border_style="red"))
    except Exception as e:
        console.print()
        console.print(Panel(f"[bold red]Error[/bold red]\n{str(e)}", border_style="red"))


def place_limit_order(order_manager: OrderManager):
    """Interactive LIMIT order placement."""
    console.print()
    console.print(Panel.fit("[bold]Place LIMIT Order[/bold]", border_style="yellow"))
    console.print()
    
    try:
        # Get order details
        symbol = Prompt.ask("Enter symbol (e.g., BTCUSDT)", default="BTCUSDT").upper()
        
        side = Prompt.ask(
            "Enter side",
            choices=["BUY", "SELL", "buy", "sell"],
            default="BUY"
        ).upper()
        
        quantity = FloatPrompt.ask("Enter quantity")
        price = FloatPrompt.ask("Enter limit price")
        
        # Display order summary
        console.print()
        table = Table(title="Order Summary", show_header=False, border_style="blue")
        table.add_column("Field", style="cyan", width=15)
        table.add_column("Value", style="white")
        
        table.add_row("Symbol", symbol)
        table.add_row("Side", f"[green]{side}[/green]" if side == "BUY" else f"[red]{side}[/red]")
        table.add_row("Order Type", "LIMIT")
        table.add_row("Quantity", str(quantity))
        table.add_row("Price", str(price))
        
        console.print(table)
        console.print()
        
        # Confirm
        if not Confirm.ask("Proceed with this order?"):
            console.print("[yellow]Order cancelled[/yellow]")
            return
        
        # Validate and place order
        order_input = validate_order_input(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=price
        )
        
        console.print()
        console.print("[yellow]Placing order...[/yellow]")
        response = order_manager.place_order(order_input)
        
        # Display result
        console.print()
        console.print(Panel(
            f"[bold green]✓ Order placed successfully![/bold green]\n"
            f"Order ID: {response.get('orderId')}\n"
            f"Status: {response.get('status')}\n"
            f"Price: {response.get('price')}",
            border_style="green"
        ))
        
    except ValueError as e:
        console.print()
        console.print(Panel(f"[bold red]Validation Error[/bold red]\n{str(e)}", border_style="red"))
    except Exception as e:
        console.print()
        console.print(Panel(f"[bold red]Error[/bold red]\n{str(e)}", border_style="red"))


def test_connection(client: BinanceClient):
    """Test API connection."""
    console.print()
    console.print(Panel.fit("[bold]Testing Connection[/bold]", border_style="yellow"))
    console.print()
    
    try:
        console.print("[yellow]Testing ping...[/yellow]")
        client.ping()
        console.print("[green]✓[/green] Ping successful")
        
        console.print("[yellow]Fetching server time...[/yellow]")
        time_resp = client.get_server_time()
        console.print(f"[green]✓[/green] Server time: {time_resp.get('serverTime')}")
        
        console.print()
        console.print(Panel("[bold green]✓ Connection test passed![/bold green]", border_style="green"))
        
    except Exception as e:
        console.print()
        console.print(Panel(f"[bold red]Connection Failed[/bold red]\n{str(e)}", border_style="red"))


def view_account_info(client: BinanceClient):
    """View account information."""
    console.print()
    console.print(Panel.fit("[bold]Account Information[/bold]", border_style="yellow"))
    console.print()
    
    try:
        console.print("[yellow]Fetching account info...[/yellow]")
        account = client.get_account_info()
        
        console.print()
        console.print(f"[cyan]Total Wallet Balance:[/cyan] {account.get('totalWalletBalance', 'N/A')} USDT")
        console.print(f"[cyan]Available Balance:[/cyan] {account.get('availableBalance', 'N/A')} USDT")
        console.print(f"[cyan]Total Position Initial Margin:[/cyan] {account.get('totalPositionInitialMargin', 'N/A')} USDT")
        
        # Show positions
        positions = [p for p in account.get('positions', []) if float(p.get('positionAmt', 0)) != 0]
        if positions:
            console.print()
            console.print("[bold]Active Positions:[/bold]")
            for pos in positions:
                console.print(f"  {pos.get('symbol')}: {pos.get('positionAmt')} @ {pos.get('entryPrice')}")
        else:
            console.print()
            console.print("[dim]No active positions[/dim]")
        
    except Exception as e:
        console.print()
        console.print(Panel(f"[bold red]Error[/bold red]\n{str(e)}", border_style="red"))


def view_open_orders(order_manager: OrderManager):
    """View open orders."""
    console.print()
    console.print(Panel.fit("[bold]Open Orders[/bold]", border_style="yellow"))
    console.print()
    
    try:
        console.print("[yellow]Fetching open orders...[/yellow]")
        orders = order_manager.get_open_orders()
        
        if not orders:
            console.print()
            console.print("[dim]No open orders[/dim]")
            return
        
        console.print()
        table = Table(title=f"Open Orders ({len(orders)})", border_style="blue")
        table.add_column("Order ID", style="cyan")
        table.add_column("Symbol", style="white")
        table.add_column("Side", style="white")
        table.add_column("Type", style="white")
        table.add_column("Price", style="white")
        table.add_column("Quantity", style="white")
        table.add_column("Status", style="green")
        
        for order in orders:
            table.add_row(
                str(order.get('orderId')),
                order.get('symbol'),
                order.get('side'),
                order.get('type'),
                order.get('price'),
                order.get('origQty'),
                order.get('status')
            )
        
        console.print(table)
        
    except Exception as e:
        console.print()
        console.print(Panel(f"[bold red]Error[/bold red]\n{str(e)}", border_style="red"))


def main():
    """Main interactive menu loop."""
    # Setup logging
    logger = setup_logging()
    
    # Display header
    display_header()
    
    # Get credentials
    console.print("[yellow]Loading API credentials...[/yellow]")
    api_key, api_secret = get_credentials()
    
    if not api_key or not api_secret:
        console.print()
        Prompt.ask("Press Enter to exit")
        return
    
    console.print("[green]✓[/green] Credentials loaded")
    console.print()
    
    # Initialize client
    console.print("[yellow]Initializing client...[/yellow]")
    try:
        client = BinanceClient(api_key, api_secret)
        order_manager = OrderManager(client)
        console.print("[green]✓[/green] Client initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize: {e}")
        console.print()
        Prompt.ask("Press Enter to exit")
        return
    
    # Main menu loop
    while True:
        console.print()
        console.print("[bold cyan]Main Menu[/bold cyan]")
        console.print()
        display_menu()
        
        choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5"], default="1")
        
        if choice == "0":
            console.print()
            console.print("[cyan]Goodbye![/cyan]")
            break
        elif choice == "1":
            place_market_order(order_manager)
        elif choice == "2":
            place_limit_order(order_manager)
        elif choice == "3":
            test_connection(client)
        elif choice == "4":
            view_account_info(client)
        elif choice == "5":
            view_open_orders(order_manager)
        
        console.print()
        Prompt.ask("Press Enter to continue")
        display_header()


if __name__ == "__main__":
    main()