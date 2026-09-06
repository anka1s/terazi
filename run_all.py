"""
run_all.py
===========
Zamanlanmış bulut görevinin (routine) her tetiklenişinde çalıştırdığı giriş
noktası. Üç sembolü (BTC, ETH, SOL) TEK bir paylaşılan RiskManager/portföy
üzerinden tek tur (`run_once`) çalıştırır, ardından portföy durumu
`portfolio_state.json`'a yazılır — bu dosya her çalıştırmadan sonra git'e
commit+push edilerek sonraki çalıştırmaya taşınır (bkz. routine talimatı).

main.py'deki TradingBot/BotConfig/DryRunExchangeClient sınıflarını olduğu
gibi kullanır; burada sadece "birden fazla sembolü paylaşılan bir risk
yöneticisiyle çalıştırma" orkestrasyonu var (main.py'nin modül docstring'i
zaten bunu öneriyordu).

Veri kaynağı olarak `data_fetcher.DataFetcher` (yfinance/Yahoo Finance)
DEĞİL, `binance_fetcher.BinanceFetcher` kullanılıyor: bazı bulut sandbox
ortamlarının egress güvenlik duvarı Yahoo Finance'i 403 ile engelliyor,
Binance'in herkese açık API'si ise kimlik doğrulama gerektirmediği için
bu tür ortamlarda daha güvenilir çalışıyor. Semboller de buna göre
doğrudan Binance formatında (BTCUSDT, ETHUSDT, SOLUSDT).
"""

from __future__ import annotations

from pathlib import Path

from binance_fetcher import BinanceFetcher
from main import BotConfig, DryRunExchangeClient, TradingBot, configure_logging
from risk_manager import RiskConfig, RiskManager

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "portfolio_state.json"

SYMBOLS = [
    BotConfig(
        data_symbol="BTCUSDT", exchange_symbol="BTCUSDT",
        asset_name="Bitcoin", news_query="Bitcoin crypto",
        period="1mo", interval="1h",
    ),
    BotConfig(
        data_symbol="ETHUSDT", exchange_symbol="ETHUSDT",
        asset_name="Ethereum", news_query="Ethereum crypto",
        period="1mo", interval="1h",
    ),
    BotConfig(
        data_symbol="SOLUSDT", exchange_symbol="SOLUSDT",
        asset_name="Solana", news_query="Solana crypto",
        period="1mo", interval="1h",
    ),
]


def main() -> None:
    configure_logging()
    risk_manager = RiskManager(RiskConfig(initial_balance=10_000), state_file=STATE_FILE)
    exchange = DryRunExchangeClient()
    data_fetcher = BinanceFetcher()

    for config in SYMBOLS:
        bot = TradingBot(config, risk_manager, exchange, data_fetcher=data_fetcher)
        bot.run_once()


if __name__ == "__main__":
    main()
