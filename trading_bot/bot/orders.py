"""Order placement logic for Binance Futures."""
import logging
from typing import Dict, Any, Optional

from bot.client import BinanceClient
from bot.validators import OrderInput


logger = logging.getLogger("trading_bot.orders")


class OrderManager:
    """Manager for placing orders on Binance Futures."""
    
    def __init__(self, client: BinanceClient):
        """
        Initialize OrderManager.
        
        Args:
            client: BinanceClient instance
        """
        self.client = client
        logger.info("OrderManager initialized")
    
    def place_order(self, order_input: OrderInput) -> Dict[str, Any]:
        """
        Place an order on Binance Futures.
        
        Args:
            order_input: Validated order input
            
        Returns:
            Order response from Binance
            
        Raises:
            Exception: On order placement failure
        """
        logger.info("=" * 80)
        logger.info("ORDER REQUEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Symbol:       {order_input.symbol}")
        logger.info(f"Side:         {order_input.side}")
        logger.info(f"Order Type:   {order_input.order_type}")
        logger.info(f"Quantity:     {order_input.quantity}")
        if order_input.price:
            logger.info(f"Price:        {order_input.price}")
        logger.info("=" * 80)
        
        # Prepare order parameters
        params = {
            'symbol': order_input.symbol,
            'side': order_input.side,
            'type': order_input.order_type,
            'quantity': order_input.quantity,
        }
        
        # Add time in force for LIMIT orders
        if order_input.order_type == "LIMIT":
            params['timeInForce'] = "GTC"  # Good Till Cancelled
            params['price'] = order_input.price
        
        try:
            logger.info(f"Placing {order_input.order_type} order...")
            response = self.client._request("POST", "/fapi/v1/order", params=params)
            
            logger.info("=" * 80)
            logger.info("ORDER RESPONSE")
            logger.info("=" * 80)
            logger.info(f"Order ID:      {response.get('orderId', 'N/A')}")
            logger.info(f"Client Order ID: {response.get('clientOrderId', 'N/A')}")
            logger.info(f"Status:        {response.get('status', 'N/A')}")
            logger.info(f"Symbol:        {response.get('symbol', 'N/A')}")
            logger.info(f"Side:          {response.get('side', 'N/A')}")
            logger.info(f"Type:          {response.get('type', 'N/A')}")
            logger.info(f"Original Qty:  {response.get('origQty', 'N/A')}")
            logger.info(f"Executed Qty:  {response.get('executedQty', 'N/A')}")
            
            if response.get('avgPrice'):
                logger.info(f"Avg Price:     {response.get('avgPrice')}")
            if response.get('price') and order_input.order_type == "LIMIT":
                logger.info(f"Limit Price:   {response.get('price')}")
            
            logger.info(f"Update Time:   {response.get('updateTime', 'N/A')}")
            logger.info("=" * 80)
            
            # Determine success/failure
            status = response.get('status', '')
            if status in ['NEW', 'FILLED', 'PARTIALLY_FILLED']:
                logger.info(f"✅ Order placed successfully! Order ID: {response.get('orderId')}")
            else:
                logger.warning(f"⚠️ Order placed but status is: {status}")
            
            return response
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("ORDER FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 80)
            raise
    
    def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Query order status.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Order information
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        logger.info(f"Querying order status for Order ID {order_id}...")
        return self.client._request("GET", "/fapi/v1/order", params=params)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Cancellation response
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        logger.info(f"Cancelling order {order_id}...")
        return self.client._request("DELETE", "/fapi/v1/order", params=params)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        logger.info(f"Fetching open orders for {symbol if symbol else 'all symbols'}...")
        return self.client._request("GET", "/fapi/v1/openOrders", params=params)