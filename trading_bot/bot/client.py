"""Binance Futures Testnet client wrapper."""
import logging
import time
import hmac
import hashlib
from typing import Dict, Any, Optional
from urllib.parse import urlencode

import requests


logger = logging.getLogger("trading_bot.client")


class BinanceClient:
    """Wrapper for Binance Futures Testnet API."""
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com"
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        })
        
        logger.info("Binance Futures Testnet client initialized")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for authenticated requests.
        
        Args:
            params: Request parameters
            
        Returns:
            Hex-encoded signature
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """
        Make API request to Binance.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.exceptions.RequestException: On API errors
        """
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        logger.debug(f"API Request: {method} {endpoint}")
        logger.debug(f"Parameters: {params}")
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=10)
            elif method == "POST":
                response = self.session.post(url, params=params, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            logger.debug(f"API Response Status: {response.status_code}")
            logger.debug(f"API Response: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def ping(self) -> Dict[str, Any]:
        """
        Test connectivity to the API.
        
        Returns:
            Empty dict if successful
        """
        logger.info("Testing API connectivity...")
        return self._request("GET", "/fapi/v1/ping", signed=False)
    
    def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time.
        
        Returns:
            Server time response
        """
        logger.info("Fetching server time...")
        return self._request("GET", "/fapi/v1/time", signed=False)
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information
        """
        logger.info("Fetching account information...")
        return self._request("GET", "/fapi/v2/account")
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Args:
            symbol: Optional symbol to filter
            
        Returns:
            Exchange information
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        logger.info(f"Fetching exchange info for {symbol if symbol else 'all symbols'}...")
        return self._request("GET", "/fapi/v1/exchangeInfo", params=params, signed=False)