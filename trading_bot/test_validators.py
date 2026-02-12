"""Unit tests for validators module."""
import pytest
from bot.validators import validate_order_input, OrderInput


def test_valid_market_order():
    """Test validation of a valid MARKET order."""
    order = validate_order_input(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.001
    )
    
    assert order.symbol == "BTCUSDT"
    assert order.side == "BUY"
    assert order.order_type == "MARKET"
    assert order.quantity == 0.001
    assert order.price is None


def test_valid_limit_order():
    """Test validation of a valid LIMIT order."""
    order = validate_order_input(
        symbol="ETHUSDT",
        side="SELL",
        order_type="LIMIT",
        quantity=0.01,
        price=3200.50
    )
    
    assert order.symbol == "ETHUSDT"
    assert order.side == "SELL"
    assert order.order_type == "LIMIT"
    assert order.quantity == 0.01
    assert order.price == 3200.50


def test_symbol_normalization():
    """Test that symbols are normalized to uppercase."""
    order = validate_order_input(
        symbol="btcusdt",
        side="buy",
        order_type="market",
        quantity=0.001
    )
    
    assert order.symbol == "BTCUSDT"
    assert order.side == "BUY"
    assert order.order_type == "MARKET"


def test_invalid_symbol():
    """Test that invalid symbols are rejected."""
    with pytest.raises(ValueError, match="Symbol must end with USDT"):
        validate_order_input(
            symbol="BTC",
            side="BUY",
            order_type="MARKET",
            quantity=0.001
        )


def test_invalid_side():
    """Test that invalid sides are rejected."""
    with pytest.raises(ValueError, match="Side must be BUY or SELL"):
        validate_order_input(
            symbol="BTCUSDT",
            side="HOLD",
            order_type="MARKET",
            quantity=0.001
        )


def test_invalid_order_type():
    """Test that invalid order types are rejected."""
    with pytest.raises(ValueError, match="Order type must be MARKET or LIMIT"):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="STOP",
            quantity=0.001
        )


def test_negative_quantity():
    """Test that negative quantities are rejected."""
    with pytest.raises(ValueError):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=-0.001
        )


def test_zero_quantity():
    """Test that zero quantities are rejected."""
    with pytest.raises(ValueError):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0
        )


def test_limit_order_without_price():
    """Test that LIMIT orders without price are rejected."""
    with pytest.raises(ValueError, match="Price is required for LIMIT orders"):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001
        )


def test_market_order_with_price():
    """Test that MARKET orders with price are rejected."""
    with pytest.raises(ValueError, match="Price should not be specified for MARKET orders"):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.001,
            price=50000
        )


def test_negative_price():
    """Test that negative prices are rejected."""
    with pytest.raises(ValueError):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001,
            price=-50000
        )


def test_zero_price():
    """Test that zero prices are rejected."""
    with pytest.raises(ValueError):
        validate_order_input(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001,
            price=0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])