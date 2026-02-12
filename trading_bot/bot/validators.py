"""Input validation functions for trading bot."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class OrderInput(BaseModel):
    """Model for order input validation."""
    
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: str = Field(..., description="Order side: BUY or SELL")
    order_type: str = Field(..., description="Order type: MARKET or LIMIT")
    quantity: float = Field(..., gt=0, description="Order quantity (must be positive)")
    price: Optional[float] = Field(None, gt=0, description="Limit price (required for LIMIT orders)")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        symbol = v.upper().strip()
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if not symbol.endswith("USDT"):
            raise ValueError("Symbol must end with USDT (e.g., BTCUSDT)")
        return symbol
    
    @field_validator('side')
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side."""
        side = v.upper().strip()
        if side not in ["BUY", "SELL"]:
            raise ValueError("Side must be BUY or SELL")
        return side
    
    @field_validator('order_type')
    @classmethod
    def validate_order_type(cls, v: str) -> str:
        """Validate order type."""
        order_type = v.upper().strip()
        if order_type not in ["MARKET", "LIMIT"]:
            raise ValueError("Order type must be MARKET or LIMIT")
        return order_type
    
    def validate_limit_price(self) -> None:
        """Validate that price is provided for LIMIT orders."""
        if self.order_type == "LIMIT" and self.price is None:
            raise ValueError("Price is required for LIMIT orders")
        if self.order_type == "MARKET" and self.price is not None:
            raise ValueError("Price should not be specified for MARKET orders")


def validate_order_input(
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: Optional[float] = None
) -> OrderInput:
    """
    Validate order input parameters.
    
    Args:
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        order_type: Order type (MARKET/LIMIT)
        quantity: Order quantity
        price: Limit price (optional, required for LIMIT orders)
        
    Returns:
        Validated OrderInput object
        
    Raises:
        ValueError: If validation fails
    """
    order = OrderInput(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price
    )
    
    # Additional validation for LIMIT orders
    order.validate_limit_price()
    
    return order