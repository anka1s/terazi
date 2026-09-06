"""
data_fetcher.py
================
Piyasa verilerini (OHLCV) çekmekten sorumlu modül.

Bu modül, yfinance üzerinden geçmiş fiyat verilerini indirir, temizler ve
diğer modüllerin (indicators.py, ai_sentiment.py, risk_manager.py, main.py)
kullanabileceği standart bir pandas.DataFrame formatına dönüştürür.

Not: Lighter borsası (veya seçilen başka bir borsa) genellikle kendi REST/WS
API'sini kullanır. Bu modüldeki `DataFetcher` sınıfı, geçmiş veri / backtest
ve teknik analiz için yfinance kullanır. Canlı emir akışı ve gerçek zamanlı
fiyatlar için ileride `fetch_live_price()` metodunu borsanızın kendi API'sine
bağlayacağız (main.py içinde entegre edilecek).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


class DataFetchError(Exception):
    """Veri çekme sırasında oluşan hatalar için özel exception."""


@dataclass
class FetchConfig:
    """Veri çekme parametrelerini tek yerde toplayan konfigürasyon nesnesi."""

    symbol: str
    period: str = "6mo"      # ör: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval: str = "1h"     # ör: 1m, 5m, 15m, 1h, 1d
    auto_adjust: bool = True


class DataFetcher:
    """
    Geçmiş fiyat verilerini indirip standart bir formata dönüştüren sınıf.

    Örnek kullanım:
        fetcher = DataFetcher()
        df = fetcher.fetch("BTC-USD", period="3mo", interval="1h")
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def fetch(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1h",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Tek bir sembol için geçmiş OHLCV verisini indirir.

        Args:
            symbol: Örn. "BTC-USD", "AAPL", "THYAO.IS"
            period: yfinance period parametresi.
            interval: yfinance interval parametresi.
            auto_adjust: Kurumsal aksiyonlara göre fiyat ayarlaması yapılsın mı.

        Returns:
            DatetimeIndex'e sahip, Open/High/Low/Close/Volume kolonlarını
            içeren temizlenmiş bir DataFrame.

        Raises:
            DataFetchError: Veri çekilemez veya boşsa.
        """
        config = FetchConfig(symbol=symbol, period=period, interval=interval, auto_adjust=auto_adjust)
        return self._fetch_with_retry(config)

    def fetch_multiple(
        self,
        symbols: list[str],
        period: str = "6mo",
        interval: str = "1h",
    ) -> dict[str, pd.DataFrame]:
        """
        Birden fazla sembol için veri çeker. Bir sembol başarısız olursa
        diğerlerini etkilemeden loglayıp devam eder.

        Returns:
            {symbol: DataFrame} şeklinde bir sözlük. Başarısız semboller
            sonuca dahil edilmez.
        """
        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                results[sym] = self.fetch(sym, period=period, interval=interval)
            except DataFetchError as exc:
                logger.warning("Sembol atlandı: %s (%s)", sym, exc)
        return results

    def _fetch_with_retry(self, config: FetchConfig) -> pd.DataFrame:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Veri çekiliyor: %s | period=%s interval=%s (deneme %d/%d)",
                    config.symbol, config.period, config.interval, attempt, self.max_retries,
                )
                raw = yf.download(
                    tickers=config.symbol,
                    period=config.period,
                    interval=config.interval,
                    auto_adjust=config.auto_adjust,
                    progress=False,
                    multi_level_index=False,
                )
                return self._clean(raw, config.symbol)
            except Exception as exc:  # yfinance ağ/parse hatalarını yakala
                last_error = exc
                logger.warning("Deneme %d başarısız: %s", attempt, exc)
                if attempt < self.max_retries:
                    backoff = min(2 ** attempt, 10)
                    time.sleep(backoff)

        raise DataFetchError(
            f"'{config.symbol}' için veri çekilemedi ({self.max_retries} deneme sonrası): {last_error}"
        )

    def _clean(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Ham yfinance çıktısını doğrular ve standart forma getirir."""
        if df is None or df.empty:
            raise DataFetchError(f"'{symbol}' için boş veri seti döndü.")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataFetchError(f"'{symbol}' verisinde eksik kolonlar: {missing}")

        df = df.copy()
        df.index.name = "Datetime"
        df = df.dropna(subset=self.REQUIRED_COLUMNS)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # Diğer modüllerin erişebilmesi için sembolü meta veri olarak ekle
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    # Hızlı manuel test: python data_fetcher.py
    fetcher = DataFetcher()
    data = fetcher.fetch("BTC-USD", period="5d", interval="1h")
    print(data.tail())
    print(f"\nToplam satır: {len(data)} | Sembol: {data.attrs.get('symbol')}")
