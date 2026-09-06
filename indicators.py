"""
indicators.py
=============
Teknik analiz göstergelerini hesaplayan modül.

`data_fetcher.py` içindeki `DataFetcher.fetch()` metodundan dönen
Open/High/Low/Close/Volume kolonlarına sahip bir DataFrame alır ve
üzerine RSI, MACD gibi indikatör kolonları ekler.

Tüm hesaplamalar yalnızca pandas/numpy ile yapılır (ta-lib gibi harici
bir bağımlılık gerekmez), böylece kurulumu basit kalır.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndicatorError(Exception):
    """İndikatör hesaplama sırasında oluşan hatalar için özel exception."""


def _validate(df: pd.DataFrame, min_rows: int) -> None:
    if "Close" not in df.columns:
        raise IndicatorError("DataFrame'de 'Close' kolonu bulunamadı.")
    if len(df) < min_rows:
        raise IndicatorError(
            f"Yetersiz veri: {len(df)} satır var, en az {min_rows} satır gerekiyor."
        )


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    RSI (Relative Strength Index) hesaplar (Wilder'ın orijinal yöntemi,
    üstel ağırlıklı ortalama - EWM ile).

    Args:
        df: Fiyat verisini içeren DataFrame.
        period: RSI periyodu (varsayılan 14).
        column: Hesaplamada kullanılacak fiyat kolonu.

    Returns:
        0-100 arası değerler alan RSI serisi. İlk `period` satır NaN olur.
    """
    _validate(df, period + 1)

    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder'ın düzleştirme yöntemi = alpha = 1/period olan EWM
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Kayıp sıfırsa (sürekli yükseliş) RSI 100 olmalı
    rsi = rsi.where(avg_loss != 0, 100)
    rsi.name = f"RSI_{period}"
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) hesaplar.

    Returns:
        Üç kolonlu bir DataFrame: MACD, Signal, Histogram.
    """
    _validate(df, slow + signal)

    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": histogram,
        },
        index=df.index,
    )


def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Basit hareketli ortalama (SMA)."""
    _validate(df, period)
    sma = df[column].rolling(window=period, min_periods=period).mean()
    sma.name = f"SMA_{period}"
    return sma


def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Üstel hareketli ortalama (EMA)."""
    _validate(df, period)
    ema = df[column].ewm(span=period, adjust=False).mean()
    ema.name = f"EMA_{period}"
    return ema


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0, column: str = "Close"
) -> pd.DataFrame:
    """Bollinger Bantları: orta (SMA), üst ve alt bantlar."""
    _validate(df, period)
    mid = df[column].rolling(window=period, min_periods=period).mean()
    std = df[column].rolling(window=period, min_periods=period).std()

    return pd.DataFrame(
        {
            "BB_Mid": mid,
            "BB_Upper": mid + num_std * std,
            "BB_Lower": mid - num_std * std,
        },
        index=df.index,
    )


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR (Average True Range) - risk_manager.py içinde stop-loss / pozisyon
    boyutlandırma için kullanılacak volatilite göstergesi.
    """
    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            raise IndicatorError(f"ATR hesaplamak için '{col}' kolonu gerekli.")
    _validate(df, period + 1)

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr.name = f"ATR_{period}"
    return atr


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Osilatör (%K ve %D).

    %K, fiyatın son `k_period` barlık aralıktaki konumunu 0-100 arasında
    ölçer; `smooth_k` ile yumuşatılır. %D, %K'nın `d_period` periyotluk
    hareketli ortalamasıdır (sinyal çizgisi).

    Returns:
        İki kolonlu DataFrame: Stoch_%K, Stoch_%D.
    """
    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            raise IndicatorError(f"Stochastic hesaplamak için '{col}' kolonu gerekli.")
    _validate(df, k_period + smooth_k + d_period)

    lowest_low = df["Low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df["High"].rolling(window=k_period, min_periods=k_period).max()

    raw_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"Stoch_%K": k, "Stoch_%D": d}, index=df.index)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ADX (Average Directional Index) - trend gücünü ölçer (yön belirtmez).
    Yan ürün olarak +DI ve -DI (yön göstergeleri) de döndürülür; bunlar
    ADX ile birlikte trend yönünü teyit etmek için kullanılabilir.

    Genel yorum: ADX < 20 zayıf/trendsiz piyasa, ADX > 25 belirgin trend.

    Returns:
        Üç kolonlu DataFrame: ADX, Plus_DI, Minus_DI.
    """
    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            raise IndicatorError(f"ADX hesaplamak için '{col}' kolonu gerekli.")
    _validate(df, period * 2)

    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {"ADX": adx, "Plus_DI": plus_di, "Minus_DI": minus_di}, index=df.index
    )


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    OBV (On-Balance Volume) - hacim akışını fiyat yönüyle ilişkilendiren
    kümülatif bir göstergedir. Fiyat/hacim uyumsuzlukları (divergence)
    tersine dönüş sinyali olarak yorumlanabilir.
    """
    if "Volume" not in df.columns:
        raise IndicatorError("OBV hesaplamak için 'Volume' kolonu gerekli.")
    _validate(df, 2)

    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (direction * df["Volume"]).cumsum()
    obv.name = "OBV"
    return obv


def add_all_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    sma_periods: tuple[int, ...] = (20, 50),
    include_bollinger: bool = True,
    include_atr: bool = True,
    include_stochastic: bool = True,
    include_adx: bool = True,
    include_obv: bool = True,
) -> pd.DataFrame:
    """
    Verilen fiyat DataFrame'ine tüm temel indikatörleri kolon olarak ekler.

    Bu fonksiyon, `main.py`'nin çağıracağı ana giriş noktasıdır: tek bir
    fonksiyon çağrısıyla RSI, MACD, SMA'lar, (opsiyonel) Bollinger ve ATR
    hesaplanıp orijinal DataFrame'e eklenir.

    NOT: `main.py` içindeki `SignalGenerator._technical_score()`, "Stoch_%K"
    kolonunu ve `latest.get("ADX")` değerini zorunlu olarak kullanır. Bu
    yüzden `include_stochastic` ve `include_adx` varsayılan olarak açık
    bırakılmalı; kapatılırsa bot hesaplama yapamaz ve sürekli HOLD döner.

    Args:
        df: `DataFetcher.fetch()` çıktısı OHLCV DataFrame'i.
        rsi_period: RSI periyodu.
        macd_params: (fast, slow, signal) üçlüsü.
        sma_periods: Eklenecek SMA periyotlarının listesi.
        include_bollinger: Bollinger bantları eklensin mi.
        include_atr: ATR eklensin mi.
        include_stochastic: Stochastic Osilatör (%K, %D) eklensin mi.
        include_adx: ADX, +DI, -DI eklensin mi.
        include_obv: OBV (On-Balance Volume) eklensin mi.

    Returns:
        İndikatör kolonları eklenmiş yeni bir DataFrame (orijinal değiştirilmez).
    """
    result = df.copy()

    result[f"RSI_{rsi_period}"] = calculate_rsi(df, period=rsi_period)

    fast, slow, signal = macd_params
    macd_df = calculate_macd(df, fast=fast, slow=slow, signal=signal)
    result = result.join(macd_df)

    for p in sma_periods:
        result[f"SMA_{p}"] = calculate_sma(df, period=p)

    if include_bollinger:
        bb_df = calculate_bollinger_bands(df)
        result = result.join(bb_df)

    if include_atr:
        result["ATR_14"] = calculate_atr(df, period=14)

    if include_stochastic:
        stoch_df = calculate_stochastic(df)
        result = result.join(stoch_df)

    if include_adx:
        adx_df = calculate_adx(df)
        result = result.join(adx_df)

    if include_obv:
        result["OBV"] = calculate_obv(df)

    # Meta veriyi (sembol vs.) koru
    result.attrs = df.attrs
    return result


if __name__ == "__main__":
    # Hızlı manuel test: sentetik veriyle (ağ gerektirmez)
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    synthetic = pd.DataFrame(
        {
            "Open": price,
            "High": price + rng.uniform(0, 1, n),
            "Low": price - rng.uniform(0, 1, n),
            "Close": price + rng.normal(0, 0.3, n),
            "Volume": rng.integers(100, 1000, n),
        },
        index=idx,
    )

    enriched = add_all_indicators(synthetic)
    print(enriched.tail())
    print("\nKolonlar:", list(enriched.columns))
