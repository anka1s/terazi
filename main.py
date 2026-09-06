"""
main.py
========
Tüm modülleri (data_fetcher, indicators, ai_sentiment, risk_manager) birbirine
bağlayan orkestrasyon katmanı.

Akış (her döngüde):
    1. Fiyat verisi çek (data_fetcher) ve indikatörleri hesapla (indicators).
    2. Haber sentiment analizi al (ai_sentiment).
    3. Teknik + sentiment sinyallerini harmanlayarak bir işlem sinyali üret
       (SignalGenerator — bu dosyada tanımlı).
    4. Açık pozisyonlar için stop-loss/take-profit kontrolü yap (risk_manager).
    5. Yeni bir sinyal varsa, risk yöneticisinden onay al ve onaylanırsa
       borsaya emir gönder (ExchangeClient soyutlaması).

Sinyal harmanlama mantığı:
    - Teknik skor: RSI + MACD + SMA trend + Stochastic'in ağırlıklı ortalaması
      (-1..+1 arası).
    - ADX, bir YÖN göstergesi olarak DEĞİL, bir "trend filtresi" olarak
      kullanılır: ADX eşiğin altındaysa (yatay/trendsiz piyasa) yeni pozisyon
      açılmaz — RSI/MACD gibi göstergeler yatay piyasalarda yanlış sinyal
      üretmeye eğilimlidir.
    - Kompozit skor = tech_weight * teknik_skor + sentiment_weight * sentiment_skor.
    - Güvenlik filtresi: Haber sentiment'i, teknik sinyale YÜKSEK güvenle ve
      GÜÇLÜ ŞEKİLDE ters yönde işaret ediyorsa, skorları harmanlayıp
      ortalamak yerine işlem tamamen ATLANIR (veto). Böylece güçlü olumsuz
      haber akışı, salt teknik göstergelerin "ezici" ağırlığıyla göz ardı
      edilmez.

Borsa entegrasyonu:
    `ExchangeClient` soyut arayüzünün iki somutlaştırması var:
      - `DryRunExchangeClient`: Gerçek para kullanmadan, günlüğe (log) yazarak
        emirleri simüle eder. VARSAYILAN ve GÜVENLİ moddur.
      - `LighterExchangeClient`: Lighter (zkLighter perpetuals, zk-rollup)
        borsasına bağlanmak için iskelet. Lighter'da her emir imzalı bir
        blockchain işlemidir (bkz. resmi `lighter-sdk` paketi ve
        https://apidocs.lighter.xyz/docs). Emir oluşturma/iptal parametreleri
        (market_index, client_order_index, order type enum'ları vb.) borsanın
        kendi dokümantasyonundaki güncel örneklerle doğrulanmadan tahmin
        edilerek yazılmamıştır — bu, gerçek para ile yanlış emir gönderme
        riskini taşır. Bu sınıf, doğru SDK kurulum/imzalama akışını gösterir
        ve tam emir mantığını eklemeniz için açıkça işaretlenmiş yerler bırakır.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data_fetcher import DataFetcher, DataFetchError
from indicators import add_all_indicators, IndicatorError
from ai_sentiment import get_symbol_sentiment, SentimentResult
from risk_manager import RiskManager, RiskConfig, Direction, TradeDecision

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Sinyal üretimi (teknik + sentiment harmanlama)
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    tech_weight: float = 0.6
    sentiment_weight: float = 0.4

    entry_threshold: float = 0.35          # |kompozit skor| bu değeri aşarsa işlem açılır
    adx_trend_threshold: float = 20.0       # ADX bu değerin altındaysa yeni işlem açılmaz

    # Sentiment veto filtresi: teknik sinyale güçlü şekilde ters düşen,
    # yüksek güvenli haber akışı işlemi tamamen iptal eder.
    veto_sentiment_confidence: float = 0.5
    veto_sentiment_score: float = 0.4       # |sentiment.score| bu değeri aşmalı


@dataclass
class Signal:
    direction: Optional[Direction]   # None = HOLD (işlem yok)
    composite_score: float
    technical_score: float
    sentiment_score: float
    confidence: float
    reason: str


class SignalGenerator:
    """Teknik indikatörler ve haber sentiment'ini harmanlayarak işlem sinyali üretir."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def _technical_score(self, row: pd.Series) -> Optional[float]:
        """
        RSI, MACD, SMA trend ve Stochastic'ten -1..+1 arası ağırlıklı bir
        kompozit teknik skor üretir. Gerekli kolonlardan biri NaN ise (ör.
        serinin başında yeterli veri birikmediyse) None döner.
        """
        required = ["RSI_14", "MACD_Hist", "ATR_14", "SMA_20", "SMA_50", "Stoch_%K"]
        if any(col not in row.index or pd.isna(row[col]) for col in required):
            return None

        rsi_score = _clip((50 - row["RSI_14"]) / 50, -1, 1)

        atr = row["ATR_14"] if row["ATR_14"] > 0 else 1e-9
        macd_score = _clip(row["MACD_Hist"] / atr, -1, 1)

        trend_score = 1.0 if row["SMA_20"] > row["SMA_50"] else -1.0

        stoch_score = _clip((50 - row["Stoch_%K"]) / 50, -1, 1)

        composite = (
            0.30 * rsi_score
            + 0.35 * macd_score
            + 0.20 * trend_score
            + 0.15 * stoch_score
        )
        return composite

    def generate(self, enriched_df: pd.DataFrame, sentiment: SentimentResult) -> Signal:
        """
        Args:
            enriched_df: `indicators.add_all_indicators()` çıktısı (en az
                bir satır, en güncel satır kullanılır).
            sentiment: `ai_sentiment.get_symbol_sentiment()` çıktısı.

        Returns:
            Signal. `direction=None` ise işlem açılmamalı (HOLD).
        """
        if enriched_df.empty:
            return Signal(None, 0.0, 0.0, sentiment.score, 0.0, "Veri seti boş.")

        latest = enriched_df.iloc[-1]
        tech_score = self._technical_score(latest)

        if tech_score is None:
            return Signal(
                None, 0.0, 0.0, sentiment.score, 0.0,
                "Yetersiz veri: indikatörler henüz hesaplanamadı (seri başlangıcı).",
            )

        adx = latest.get("ADX")
        trend_ok = adx is not None and not pd.isna(adx) and adx >= self.config.adx_trend_threshold

        composite = (
            self.config.tech_weight * tech_score
            + self.config.sentiment_weight * sentiment.score
        )

        if not trend_ok:
            adx_display = f"{adx:.1f}" if adx is not None and not pd.isna(adx) else "N/A"
            return Signal(
                None, composite, tech_score, sentiment.score, 0.0,
                f"ADX ({adx_display}) trend eşiğinin altında "
                f"({self.config.adx_trend_threshold}) — piyasa yatay, yeni işlem açılmıyor.",
            )

        tech_sign = _sign(tech_score)
        sentiment_sign = _sign(sentiment.score)
        sentiment_vetoes = (
            sentiment_sign != 0
            and sentiment_sign != tech_sign
            and sentiment.confidence >= self.config.veto_sentiment_confidence
            and abs(sentiment.score) >= self.config.veto_sentiment_score
        )

        if sentiment_vetoes:
            return Signal(
                None, composite, tech_score, sentiment.score, 0.0,
                f"Haber sentiment'i ({sentiment.score:+.2f}, güven={sentiment.confidence:.2f}) "
                f"teknik sinyale ({tech_score:+.2f}) güçlü şekilde ters düşüyor — işlem atlandı.",
            )

        confidence = min(1.0, abs(composite))

        if composite >= self.config.entry_threshold:
            return Signal(Direction.LONG, composite, tech_score, sentiment.score, confidence,
                           f"Kompozit skor {composite:+.2f} >= eşik {self.config.entry_threshold} -> LONG")
        if composite <= -self.config.entry_threshold:
            return Signal(Direction.SHORT, composite, tech_score, sentiment.score, confidence,
                           f"Kompozit skor {composite:+.2f} <= -eşik -> SHORT")

        return Signal(
            None, composite, tech_score, sentiment.score, confidence,
            f"Kompozit skor {composite:+.2f}, eşik {self.config.entry_threshold} altında -> HOLD",
        )


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sign(value: float, tolerance: float = 1e-9) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Borsa entegrasyonu (Exchange abstraction)
# ---------------------------------------------------------------------------


class ExchangeClient(ABC):
    """Borsaya emir gönderme/fiyat okuma işlemleri için soyut arayüz."""

    @abstractmethod
    def get_current_price(self, symbol: str, fallback: Optional[float] = None) -> float:
        ...

    @abstractmethod
    def open_order(self, symbol: str, direction: Direction, size: float) -> dict:
        ...

    @abstractmethod
    def close_order(self, symbol: str, size: float) -> dict:
        ...


class DryRunExchangeClient(ExchangeClient):
    """
    Gerçek para kullanmadan emirleri simüle eden, VARSAYILAN ve GÜVENLİ mod.
    Fiyat verisi olarak her zaman `data_fetcher`'dan gelen son kapanış
    fiyatını (fallback) kullanır ve emirlerin anında, kaymasız (slippage'siz)
    dolduğunu varsayar.
    """

    def get_current_price(self, symbol: str, fallback: Optional[float] = None) -> float:
        if fallback is None:
            raise ValueError("DryRunExchangeClient için 'fallback' fiyatı zorunludur.")
        return fallback

    def open_order(self, symbol: str, direction: Direction, size: float) -> dict:
        logger.info("[DRY-RUN] EMİR AÇ: %s %s | boyut=%.6f", symbol, direction.value, size)
        return {"status": "filled", "mode": "dry_run", "symbol": symbol, "size": size}

    def close_order(self, symbol: str, size: float) -> dict:
        logger.info("[DRY-RUN] EMİR KAPAT: %s | boyut=%.6f", symbol, size)
        return {"status": "closed", "mode": "dry_run", "symbol": symbol, "size": size}


class LighterExchangeClient(ExchangeClient):
    """
    Lighter (zkLighter perpetuals, zk-rollup DEX) için canlı emir istemcisi
    İSKELETİ.

    Lighter'da her yazma işlemi (emir açma/iptal) imzalı bir blockchain
    işlemidir; okuma işlemleri normal HTTP GET'tir. Resmi Python SDK'sı:

        pip install lighter-sdk

    Kurulum (resmi dokümantasyondan doğrulanmış akış):
        import lighter
        client = lighter.SignerClient(
            url=BASE_URL,                                  # ör. https://mainnet.zklighter.elliot.ai
            api_private_keys={API_KEY_INDEX: PRIVATE_KEY},
            account_index=ACCOUNT_INDEX,
        )

    ÖNEMLİ: Emir oluşturma/iptal etme metodlarının tam parametreleri
    (market_index, client_order_index, order type/time-in-force enum'ları
    vb.) burada TAHMİN EDİLEREK yazılmamıştır — bu gerçek parayla işlem yapan
    bir borsa, yanlış varsayılmış bir parametre gerçek zararla sonuçlanabilir.
    Aşağıdaki metodları doldurmadan önce mutlaka şunlara bakın:
        - https://apidocs.lighter.xyz/docs (resmi dokümantasyon)
        - https://github.com/elliottech/lighter-python/tree/main/examples
          (özellikle "Create / modify / cancel an order" örnekleri)
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get(
            "LIGHTER_BASE_URL", "https://mainnet.zklighter.elliot.ai"
        )
        self.account_index = os.environ.get("LIGHTER_ACCOUNT_INDEX")
        self.api_key_index = os.environ.get("LIGHTER_API_KEY_INDEX")
        self.private_key = os.environ.get("LIGHTER_PRIVATE_KEY")
        self._client = None

        if not all([self.account_index, self.api_key_index, self.private_key]):
            logger.warning(
                "LIGHTER_ACCOUNT_INDEX / LIGHTER_API_KEY_INDEX / LIGHTER_PRIVATE_KEY "
                "ortam değişkenleri eksik. LighterExchangeClient yalnızca bunlar "
                "tanımlandığında gerçek emir gönderebilir."
            )

    def _get_client(self):
        if self._client is None:
            import lighter  # yalnızca gerektiğinde import edilir (pip install lighter-sdk)

            self._client = lighter.SignerClient(
                url=self.base_url,
                api_private_keys={int(self.api_key_index): self.private_key},
                account_index=int(self.account_index),
            )
        return self._client

    def get_current_price(self, symbol: str, fallback: Optional[float] = None) -> float:
        # TODO: lighter.OrderApi üzerinden ilgili market_index için gerçek
        # order book / son işlem fiyatını çekin (bkz. resmi dokümantasyon:
        # OrderApi.order_book_details / recent_trades). Yanlış bir varsayım
        # yapmamak için burada yfinance/data_fetcher'dan gelen fiyata
        # (fallback) düşülüyor.
        if fallback is not None:
            logger.warning(
                "LighterExchangeClient.get_current_price henüz canlı orderbook'a "
                "bağlanmıyor; data_fetcher'dan gelen son fiyat (%.4f) kullanılıyor.",
                fallback,
            )
            return fallback
        raise NotImplementedError(
            "Canlı fiyat için lighter.OrderApi entegrasyonunu tamamlayın "
            "(bkz. https://apidocs.lighter.xyz/docs)."
        )

    def open_order(self, symbol: str, direction: Direction, size: float) -> dict:
        raise NotImplementedError(
            "Lighter üzerinde gerçek emir açma, resmi SDK'nın "
            "SignerClient.create_order() (veya güncel dokümantasyondaki "
            "eşdeğeri) ile imzalı bir işlem gerektirir. Doğru market_index, "
            "client_order_index ve emir parametreleri için "
            "https://github.com/elliottech/lighter-python/tree/main/examples "
            "içindeki 'Create / modify / cancel an order' örneğine bakıp "
            "burayı doldurun."
        )

    def close_order(self, symbol: str, size: float) -> dict:
        raise NotImplementedError(
            "Lighter üzerinde pozisyon kapatma da imzalı bir emir işlemidir; "
            "open_order() ile aynı şekilde resmi SDK örneklerine bakarak "
            "doldurulmalıdır."
        )


# ---------------------------------------------------------------------------
# Bot konfigürasyonu ve orkestrasyon
# ---------------------------------------------------------------------------


@dataclass
class BotConfig:
    data_symbol: str            # yfinance sembolü, ör. "BTC-USD"
    exchange_symbol: str        # borsadaki işlem sembolü (Lighter'da market adı/index'i)
    asset_name: str             # sentiment analizi bağlamı için görünen ad, ör. "Bitcoin"
    news_query: Optional[str] = None   # None ise asset_name kullanılır
    period: str = "1mo"
    interval: str = "1h"
    poll_interval_seconds: int = 300    # canlı döngüde iki analiz arası bekleme


class TradingBot:
    """
    Tüm modülleri birbirine bağlayan orkestrasyon sınıfı. Tek bir sembol
    üzerinde çalışır; birden fazla sembol için birden fazla `TradingBot`
    örneği oluşturup ayrı ayrı çalıştırın (ortak bir `RiskManager` paylaşarak
    portföy genelinde risk limitlerini tek yerden yönetebilirsiniz).
    """

    def __init__(
        self,
        config: BotConfig,
        risk_manager: RiskManager,
        exchange_client: ExchangeClient,
        signal_generator: Optional[SignalGenerator] = None,
        data_fetcher: Optional[DataFetcher] = None,
    ):
        self.config = config
        self.risk_manager = risk_manager
        self.exchange_client = exchange_client
        self.signal_generator = signal_generator or SignalGenerator(StrategyConfig())
        self.data_fetcher = data_fetcher or DataFetcher()

    def run_once(self) -> None:
        """Tam bir analiz + karar döngüsünü bir kez çalıştırır."""
        symbol = self.config.exchange_symbol

        try:
            df = self.data_fetcher.fetch(
                self.config.data_symbol, period=self.config.period, interval=self.config.interval
            )
            enriched = add_all_indicators(df)
        except (DataFetchError, IndicatorError) as exc:
            logger.error("Veri/indikatör hatası (%s), bu döngü atlanıyor: %s", symbol, exc)
            return

        current_price = enriched["Close"].iloc[-1]

        sentiment = get_symbol_sentiment(
            asset_name=self.config.asset_name,
            query=self.config.news_query or self.config.asset_name,
        )

        signal = self.signal_generator.generate(enriched, sentiment)
        logger.info(
            "[%s] Sinyal: yön=%s kompozit=%+.2f teknik=%+.2f sentiment=%+.2f(%s) | %s",
            symbol,
            signal.direction.value if signal.direction else "HOLD",
            signal.composite_score, signal.technical_score, signal.sentiment_score,
            sentiment.method, signal.reason,
        )

        # 1) Açık pozisyonlar için çıkış kontrolü (stop-loss / take-profit)
        closed_trades = self.risk_manager.check_exits({symbol: current_price})
        for trade in closed_trades:
            self.exchange_client.close_order(symbol, trade.size)

        # 2) Yeni sinyal varsa risk onayından geçir ve (onaylanırsa) emri gönder
        if signal.direction is not None:
            atr = enriched["ATR_14"].iloc[-1]
            decision: TradeDecision = self.risk_manager.evaluate_trade(
                symbol, signal.direction, entry_price=current_price, atr=atr
            )
            if decision.approved:
                self.exchange_client.open_order(symbol, signal.direction, decision.size)
                self.risk_manager.open_position(symbol, signal.direction, current_price, decision)
            else:
                logger.info("[%s] İşlem risk yöneticisi tarafından reddedildi: %s", symbol, decision.reason)

        logger.info("[%s] Portföy özeti: %s", symbol, self.risk_manager.portfolio_summary())

    def run_forever(self) -> None:
        """
        Sürekli çalışan döngü. Bir döngüdeki beklenmeyen bir hata TÜM botu
        durdurmamalı — loglanır ve bir sonraki döngüde devam edilir.
        """
        logger.info(
            "Bot başlatıldı: %s (veri sembolü=%s, döngü aralığı=%ds)",
            self.config.exchange_symbol, self.config.data_symbol, self.config.poll_interval_seconds,
        )
        while True:
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - bot canlıyken beklenmeyen hiçbir hata çökmeye sebep olmamalı
                logger.exception("Döngüde beklenmeyen hata, bir sonraki döngüde devam edilecek.")
            time.sleep(self.config.poll_interval_seconds)


if __name__ == "__main__":
    configure_logging()

    bot_config = BotConfig(
        data_symbol="BTC-USD",
        exchange_symbol="BTC-USD",
        asset_name="Bitcoin",
        news_query="Bitcoin crypto",
        period="1mo",
        interval="1h",
        poll_interval_seconds=300,
    )

    risk_manager = RiskManager(RiskConfig(initial_balance=10_000))
    exchange_client = DryRunExchangeClient()  # GÜVENLİ VARSAYILAN — gerçek emir göndermez

    bot = TradingBot(bot_config, risk_manager, exchange_client)

    # Tek seferlik test için: bot.run_once()
    # Canlı/sürekli çalıştırmak için: bot.run_forever()
    bot.run_once()
