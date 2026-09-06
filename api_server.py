"""
api_server.py
==============
terazi.html arayüzünü GERÇEK Python botuna (data_fetcher, indicators,
main.SignalGenerator, risk_manager, ai_sentiment) bağlayan yerel API
sunucusu.

Önceden terazi.html tamamen bağımsızdı: kendi içinde JS ile indikatörleri
yeniden hesaplıyor, fiyatı tarayıcıdan doğrudan Binance'ten, sentiment'i
doğrudan (anahtarsız, çalışmayan) bir Anthropic API çağrısından alıyor ve
portföyü `window.storage` (yalnızca Claude Artifacts ortamında var olan,
normal tarayıcıda mevcut olmayan bir API) ile saklıyordu. Bu sunucu, o iş
mantığının tamamını gerçek Python modüllerine taşır; tarayıcı sadece bu
API'den JSON okuyup gösteren ince bir istemci haline gelir.

Çalıştırma:
    pip install -r requirements.txt
    uvicorn api_server:app --reload --port 8000

Sonra tarayıcıda http://localhost:8000/ adresini açın.

Not: Sentiment analizi için gerçek Claude API kullanmak isterseniz
ANTHROPIC_API_KEY ortam değişkenini tanımlayın; tanımlı değilse otomatik
olarak basit kelime sözlüğü yöntemine düşülür (bkz. ai_sentiment.py).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_sentiment import SentimentResult, analyze_headlines
from data_fetcher import DataFetchError, DataFetcher
from indicators import IndicatorError, add_all_indicators
from main import Signal, SignalGenerator, StrategyConfig
from risk_manager import Direction, RiskConfig, RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "portfolio_state.json"

SYMBOL_MAP = {
    "BTCUSDT": {"yf": "BTC-USD", "asset": "Bitcoin", "news": "Bitcoin crypto"},
    "ETHUSDT": {"yf": "ETH-USD", "asset": "Ethereum", "news": "Ethereum crypto"},
    "SOLUSDT": {"yf": "SOL-USD", "asset": "Solana", "news": "Solana crypto"},
}

app = FastAPI(title="Terazi API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

data_fetcher = DataFetcher()
signal_generator = SignalGenerator(StrategyConfig())
risk_manager = RiskManager(RiskConfig(initial_balance=10_000), state_file=STATE_FILE)

# Sunucu süreci içinde tutulan basit durum (tek kullanıcılı yerel araç
# olduğu için veritabanı gerekmiyor): her sembolün son bilinen fiyatı ve
# en son analiz edilen sentiment sonucu.
last_prices: dict[str, float] = {}
last_sentiment: dict[str, SentimentResult] = {}


def _symbol_info(symbol: str) -> dict:
    info = SYMBOL_MAP.get(symbol)
    if not info:
        raise HTTPException(400, f"Bilinmeyen sembol: {symbol}")
    return info


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _signal_to_dict(signal: Signal) -> dict:
    return {
        "direction": signal.direction.value if signal.direction else None,
        "composite": signal.composite_score,
        "tech": signal.technical_score,
        "sentiment": signal.sentiment_score,
        "confidence": signal.confidence,
        "reason": signal.reason,
    }


def _sentiment_to_dict(sentiment: SentimentResult) -> dict:
    return {
        "score": sentiment.score,
        "label": sentiment.label,
        "confidence": sentiment.confidence,
        "method": sentiment.method,
        "summary": sentiment.summary,
    }


def _trade_to_dict(trade) -> dict:
    return {
        "symbol": trade.symbol,
        "direction": trade.direction.value,
        "entryPrice": trade.entry_price,
        "exitPrice": trade.exit_price,
        "size": trade.size,
        "pnl": trade.pnl,
        "pnlPct": trade.pnl_pct,
        "reason": trade.reason,
        "closedAt": trade.closed_at,
    }


def _compute(symbol: str):
    """Fiyat çek + indikatör hesapla + son sentiment ile sinyal üret. Tüm endpoint'ler bunu paylaşır."""
    info = _symbol_info(symbol)
    try:
        df = data_fetcher.fetch(info["yf"], period="1mo", interval="1h")
        enriched = add_all_indicators(df)
    except (DataFetchError, IndicatorError) as exc:
        raise HTTPException(502, f"Veri/indikatör hatası: {exc}") from exc

    price = float(enriched["Close"].iloc[-1])
    atr = _safe_float(enriched["ATR_14"].iloc[-1]) or 0.0
    last_prices[symbol] = price

    sentiment = last_sentiment.get(symbol) or SentimentResult(0.0, "neutral", 0.0, "—", None)
    signal = signal_generator.generate(enriched, sentiment)
    return enriched, price, atr, sentiment, signal


@app.get("/api/signal")
def get_signal(symbol: str = Query("BTCUSDT")):
    enriched, price, _atr, sentiment, signal = _compute(symbol)

    closed = risk_manager.check_exits(last_prices)

    candles = [
        {
            "time": int(ts.timestamp() * 1000),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "rsi": _safe_float(row.get("RSI_14")),
            "macd": _safe_float(row.get("MACD")),
            "macdSignal": _safe_float(row.get("MACD_Signal")),
            "macdHist": _safe_float(row.get("MACD_Hist")),
            "sma20": _safe_float(row.get("SMA_20")),
            "sma50": _safe_float(row.get("SMA_50")),
            "atr": _safe_float(row.get("ATR_14")),
            "stochK": _safe_float(row.get("Stoch_%K")),
            "adx": _safe_float(row.get("ADX")),
            "obv": _safe_float(row.get("OBV")),
        }
        for ts, row in enriched.tail(200).iterrows()
    ]

    return {
        "symbol": symbol,
        "live": True,
        "price": price,
        "candles": candles,
        "signal": _signal_to_dict(signal),
        "sentiment": _sentiment_to_dict(sentiment),
        "closedTrades": [_trade_to_dict(t) for t in closed],
        "portfolio": risk_manager.portfolio_summary(),
    }


@app.get("/api/preview")
def get_preview(symbol: str = Query("BTCUSDT")):
    """Bir pozisyon açmadan önce risk yöneticisinin onaylayıp onaylamayacağını (boyut/stop/hedef ile) gösterir."""
    enriched, price, atr, _sentiment, signal = _compute(symbol)
    if signal.direction is None:
        return {"approved": False, "reason": "Şu an yürütülecek bir AL/SAT sinyali yok."}

    decision = risk_manager.evaluate_trade(symbol, signal.direction, entry_price=price, atr=atr)
    return {
        "approved": decision.approved,
        "reason": decision.reason,
        "direction": signal.direction.value,
        "size": decision.size,
        "stopLoss": decision.stop_loss,
        "takeProfit": decision.take_profit,
        "leverage": decision.leverage,
        "liquidationPrice": decision.liquidation_price,
    }


class SentimentRequest(BaseModel):
    symbol: str
    headlines: str


@app.post("/api/sentiment")
def post_sentiment(req: SentimentRequest):
    info = _symbol_info(req.symbol)
    result = analyze_headlines(info["asset"], req.headlines)
    last_sentiment[req.symbol] = result
    return _sentiment_to_dict(result)


class ExecuteRequest(BaseModel):
    symbol: str


@app.post("/api/execute")
def post_execute(req: ExecuteRequest):
    enriched, price, atr, _sentiment, signal = _compute(req.symbol)
    if signal.direction is None:
        raise HTTPException(400, "Şu an yürütülecek bir AL/SAT sinyali yok.")

    decision = risk_manager.evaluate_trade(req.symbol, signal.direction, entry_price=price, atr=atr)
    if not decision.approved:
        raise HTTPException(400, f"Risk yöneticisi reddetti: {decision.reason}")

    risk_manager.open_position(req.symbol, signal.direction, price, decision)
    return {
        "opened": True,
        "direction": signal.direction.value,
        "size": decision.size,
        "stopLoss": decision.stop_loss,
        "takeProfit": decision.take_profit,
        "leverage": decision.leverage,
        "liquidationPrice": decision.liquidation_price,
    }


@app.get("/api/portfolio")
def get_portfolio():
    return {
        "summary": risk_manager.portfolio_summary(),
        "openPositions": [
            {
                "symbol": sym,
                "direction": p.direction.value,
                "entryPrice": p.entry_price,
                "size": p.size,
                "stopLoss": p.stop_loss,
                "takeProfit": p.take_profit,
                "leverage": p.leverage,
                "liquidationPrice": p.liquidation_price,
                "lastPrice": last_prices.get(sym, p.entry_price),
            }
            for sym, p in risk_manager.open_positions.items()
        ],
        "tradeHistory": [_trade_to_dict(t) for t in risk_manager.trade_history[-50:]],
    }


@app.post("/api/portfolio/reset")
def post_reset_portfolio():
    risk_manager.reset_portfolio()
    return {"ok": True}


@app.post("/api/portfolio/reset-halt")
def post_reset_halt():
    risk_manager.reset_halt()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Statik dosyalar (terazi.html)
# ---------------------------------------------------------------------------

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "terazi.html")
