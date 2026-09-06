"""
binance_fetcher.py
====================
Binance'in herkese açık REST API'sinden kripto OHLCV verisi çeker.

`data_fetcher.DataFetcher.fetch()` ile aynı formatta (Open/High/Low/Close/
Volume kolonlu, DatetimeIndex'li) bir DataFrame üretir, böylece
`indicators.add_all_indicators()` ve `main.TradingBot` ile doğrudan
uyumludur (aynı arayüz: `.fetch(symbol, period=..., interval=...)`).

yfinance/Yahoo Finance bazı ağ ortamlarında (ör. kısıtlı egress politikalı
bulut sandbox'ları — bu proje bunu zamanlanmış bulut görevinde 403 ile
yaşadı) erişilemez olabiliyor. Binance'in herkese açık kline API'si kimlik
doğrulama gerektirmez ve kripto sembolleri için data_fetcher.DataFetcher'a
tercih edilen bir alternatiftir.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com/api/v3/klines"


class BinanceFetchError(Exception):
    """Binance'ten veri çekme sırasında oluşan hatalar için özel exception."""


class BinanceFetcher:
    """
    Örnek kullanım:
        fetcher = BinanceFetcher()
        df = fetcher.fetch("BTCUSDT", interval="1h")
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, max_retries: int = 3, timeout: float = 10.0):
        self.max_retries = max_retries
        self.timeout = timeout

    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Args:
            symbol: Binance sembolü, ör. "BTCUSDT".
            period: DataFetcher ile arayüz uyumluluğu için kabul edilir,
                kullanılmaz (Binance'te periyot yerine `limit` ile mum
                sayısı belirtilir).
            interval: Binance kline aralığı (1m, 5m, 15m, 1h, 4h, 1d, ...).
            limit: Çekilecek mum sayısı (Binance maksimum 1000 destekler).
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Binance'ten veri çekiliyor: %s | interval=%s limit=%d (deneme %d/%d)",
                    symbol, interval, limit, attempt, self.max_retries,
                )
                resp = requests.get(
                    BASE_URL,
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json()
                return self._to_dataframe(raw, symbol)
            except Exception as exc:  # ağ/parse hatalarını yakala
                last_error = exc
                logger.warning("Deneme %d başarısız: %s", attempt, exc)

        raise BinanceFetchError(
            f"'{symbol}' için Binance'ten veri çekilemedi ({self.max_retries} deneme sonrası): {last_error}"
        )

    def _to_dataframe(self, raw: list, symbol: str) -> pd.DataFrame:
        if not raw:
            raise BinanceFetchError(f"'{symbol}' için boş veri seti döndü.")

        df = pd.DataFrame(raw, columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
        ])
        df["Datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("Datetime")
        for col in self.REQUIRED_COLUMNS:
            df[col] = df[col].astype(float)
        df = df[self.REQUIRED_COLUMNS]
        df.attrs["symbol"] = symbol
        return df
