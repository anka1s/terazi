"""
ai_sentiment.py
================
Haber başlıklarından sentiment (duyarlılık) skoru üretir.

main.py'nin `from ai_sentiment import get_symbol_sentiment, SentimentResult`
importu bu modülü bekliyordu ama dosya mevcut değildi.

Öncelik sırası:
  1. Anthropic Claude API (ANTHROPIC_API_KEY ortam değişkeni tanımlıysa).
  2. Basit kelime sözlüğü tabanlı yedek yöntem (API yoksa veya başarısız
     olursa) — terazi.html'deki JS yedek yöntemiyle aynı kelime listeleri.

ÖNEMLİ: terazi.html'in eski sürümü tarayıcıdan doğrudan api.anthropic.com'a
istek atıyordu; bu güvenli değildir (API anahtarı asla istemci koduna
konulmamalı) ve zaten CORS/kimlik doğrulama eksikliği yüzünden her zaman
başarısız olup sessizce yedek yönteme düşüyordu. Bu modül aynı analizi
SUNUCU tarafında (api_server.py üzerinden) yapar; anahtar yalnızca burada,
ortam değişkeninde tutulur ve tarayıcıya asla gönderilmez.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

POS_WORDS = [
    "surge", "rally", "adoption", "bullish", "record", "growth", "soar",
    "breakout", "gain", "upgrade", "approval", "inflow",
]
NEG_WORDS = [
    "crash", "plunge", "bearish", "sell-off", "selloff", "decline", "warn",
    "correction", "hack", "ban", "lawsuit", "outflow", "downgrade", "fraud",
]


@dataclass
class SentimentResult:
    score: float             # -1.0 .. 1.0
    label: str                # "positive" | "negative" | "neutral"
    confidence: float          # 0.0 .. 1.0
    method: str
    summary: Optional[str] = None


def _fallback_lexicon_sentiment(text: str) -> SentimentResult:
    lower = text.lower()
    pos = sum(1 for w in POS_WORDS if w in lower)
    neg = sum(1 for w in NEG_WORDS if w in lower)
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / total
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    confidence = 0.0 if total == 0 else min(1.0, total * 0.2)
    return SentimentResult(
        score=score, label=label, confidence=confidence,
        method="basit sözlük (yedek)", summary=None,
    )


def _claude_sentiment(asset_name: str, headlines_text: str) -> SentimentResult:
    import anthropic  # pip install anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY ortam değişkeni tanımlı değil.")

    system_prompt = (
        "Sen bir algoritmik trading sisteminin haber duyarlılık analiz motorusun. "
        f"Sana verilen haber başlıklarını '{asset_name}' varlığı açısından değerlendir. "
        "SADECE aşağıdaki JSON formatında yanıt ver, başka hiçbir metin, markdown kod "
        "bloğu veya açıklama ekleme: "
        '{"score": <-1.0 ile 1.0 arasında float>, "label": "positive"|"negative"|"neutral", '
        '"confidence": <0.0 ile 1.0 arasında float>, "summary": "<tek cümlelik kısa Türkçe özet>"}'
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": headlines_text}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    cleaned = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    parsed = json.loads(cleaned)
    if "score" not in parsed or "confidence" not in parsed:
        raise ValueError("Claude yanıtında eksik alan (score/confidence).")

    return SentimentResult(
        score=float(parsed["score"]),
        label=parsed.get("label", "neutral"),
        confidence=float(parsed["confidence"]),
        method="Claude API",
        summary=parsed.get("summary"),
    )


def analyze_headlines(asset_name: str, headlines_text: str) -> SentimentResult:
    """Kullanıcının elle girdiği/sağladığı başlık metnini analiz eder (ör. web arayüzünden)."""
    text = (headlines_text or "").strip()
    if not text:
        return SentimentResult(0.0, "neutral", 0.0, "veri yok", None)
    try:
        return _claude_sentiment(asset_name, text)
    except Exception as exc:
        logger.warning("Claude API sentiment analizi başarısız, sözlük yedeğine geçiliyor: %s", exc)
        return _fallback_lexicon_sentiment(text)


def _fetch_google_news_headlines(query: str, max_items: int = 10) -> list[str]:
    """
    API anahtarı gerektirmeyen Google News RSS aramasından başlık çeker.
    Ağ hatasında (veya format değişikliğinde) sessizce boş liste döner —
    çağıran taraf bunu "haber bulunamadı" olarak yorumlar.
    """
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        titles = [item.findtext("title") for item in root.iter("item")]
        return [t for t in titles if t][:max_items]
    except Exception as exc:
        logger.warning("Google News RSS'den haber çekilemedi (%s): %s", query, exc)
        return []


def get_symbol_sentiment(asset_name: str, query: str) -> SentimentResult:
    """
    main.py'nin `TradingBot.run_once()` içinde çağırdığı asıl giriş noktası:
    verilen sorguyla ilgili güncel haber başlıklarını (Google News RSS, API
    anahtarı gerekmez) çeker ve bunları analiz eder.
    """
    headlines = _fetch_google_news_headlines(query)
    if not headlines:
        return SentimentResult(0.0, "neutral", 0.0, "haber bulunamadı", None)
    return analyze_headlines(asset_name, "\n".join(headlines))
