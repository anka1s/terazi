"""
risk_manager.py
================
Kağıt üzerinde (paper trading) risk yönetimi ve portföy takibi.

main.py'nin `from risk_manager import RiskManager, RiskConfig, Direction,
TradeDecision` importu bu modülü bekliyordu ama dosya mevcut değildi — bu
yüzden main.py hiç çalıştırılamıyordu. Buradaki mantık, terazi.html
içindeki JS risk motoruyla (RISK_CONFIG / evaluateTrade / openPosition /
checkExits) birebir aynıdır; böylece tarayıcı arayüzü ile botun kendi
hesapladığı sonuçlar tutarlı kalır.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class RiskConfig:
    initial_balance: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    stop_loss_atr_multiplier: float = 2.0
    min_rr: float = 2.0
    max_leverage: float = 3.0
    # İzole marjda basitleştirilmiş likidasyon mesafesi ≈ giriş_fiyatı / kaldıraç.
    # Stop-loss, bu mesafenin bu oranından daha yakınında olmalı; aksi halde
    # pozisyon stop tetiklenmeden ÖNCE likide olabilir. 0.8 = likidasyon
    # mesafesinin en fazla %80'i kadar bir stop mesafesine izin ver.
    liquidation_safety_buffer: float = 0.8
    max_open_positions: int = 5
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_consecutive_losses: int = 5


@dataclass
class TradeDecision:
    approved: bool
    reason: str
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    leverage: float = 0.0
    liquidation_price: Optional[float] = None


@dataclass
class Position:
    symbol: str
    direction: Direction
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    leverage: float
    liquidation_price: Optional[float] = None
    opened_at: float = field(default_factory=time.time)


@dataclass
class ClosedTrade:
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    reason: str
    closed_at: float = field(default_factory=time.time)


class RiskManager:
    """
    Tek bir portföy üzerinde risk kurallarını uygulayan kağıt üzerinde (paper)
    risk yöneticisi. `state_file` verilirse portföy durumu (equity, açık
    pozisyonlar, işlem geçmişi) JSON olarak diske yazılır/okunur, böylece
    `api_server.py` yeniden başlatılsa bile portföy kaybolmaz.
    """

    def __init__(self, config: RiskConfig, state_file: Optional[str | Path] = None):
        self.config = config
        self.state_file = Path(state_file) if state_file else None
        self.equity = config.initial_balance
        self.peak_equity = config.initial_balance
        self.day_start_equity = config.initial_balance
        self.day_date = date.today().isoformat()
        self.open_positions: dict[str, Position] = {}
        self.trade_history: list[ClosedTrade] = []
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason: Optional[str] = None

        if self.state_file and self.state_file.exists():
            self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Portföy durumu okunamadı, varsayılan durumla devam ediliyor.")
            return

        self.equity = data.get("equity", self.equity)
        self.peak_equity = data.get("peak_equity", self.peak_equity)
        self.day_start_equity = data.get("day_start_equity", self.day_start_equity)
        self.day_date = data.get("day_date", self.day_date)
        self.consecutive_losses = data.get("consecutive_losses", 0)
        self.trading_halted = data.get("trading_halted", False)
        self.halt_reason = data.get("halt_reason")
        self.open_positions = {
            sym: Position(
                symbol=sym,
                direction=Direction(p["direction"]),
                entry_price=p["entry_price"],
                size=p["size"],
                stop_loss=p["stop_loss"],
                take_profit=p["take_profit"],
                leverage=p["leverage"],
                liquidation_price=p.get("liquidation_price"),
                opened_at=p.get("opened_at", time.time()),
            )
            for sym, p in data.get("open_positions", {}).items()
        }
        self.trade_history = [
            ClosedTrade(
                symbol=t["symbol"],
                direction=Direction(t["direction"]),
                entry_price=t["entry_price"],
                exit_price=t["exit_price"],
                size=t["size"],
                pnl=t["pnl"],
                pnl_pct=t["pnl_pct"],
                reason=t["reason"],
                closed_at=t.get("closed_at", time.time()),
            )
            for t in data.get("trade_history", [])
        ]

    def _save(self) -> None:
        if not self.state_file:
            return
        data = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "day_date": self.day_date,
            "consecutive_losses": self.consecutive_losses,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "open_positions": {
                sym: {
                    "direction": p.direction.value,
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "leverage": p.leverage,
                    "liquidation_price": p.liquidation_price,
                    "opened_at": p.opened_at,
                }
                for sym, p in self.open_positions.items()
            },
            "trade_history": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "reason": t.reason,
                    "closed_at": t.closed_at,
                }
                for t in self.trade_history
            ],
        }
        self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ breakers
    def _roll_day(self) -> None:
        today = date.today().isoformat()
        if today != self.day_date:
            self.day_date = today
            self.day_start_equity = self.equity

    def drawdown_pct(self) -> float:
        return 0.0 if self.peak_equity <= 0 else (self.peak_equity - self.equity) / self.peak_equity

    def daily_pnl_pct(self) -> float:
        return 0.0 if self.day_start_equity <= 0 else (self.equity - self.day_start_equity) / self.day_start_equity

    def _update_breakers(self) -> None:
        if self.trading_halted:
            return
        if self.daily_pnl_pct() <= -self.config.max_daily_loss_pct:
            self.trading_halted = True
            self.halt_reason = f"Günlük zarar limiti aşıldı ({self.daily_pnl_pct() * 100:.2f}%)"
            return
        if self.drawdown_pct() >= self.config.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = (
                f"Maksimum drawdown limiti aşıldı ({self.drawdown_pct() * 100:.2f}%) — manuel reset gerekli."
            )
            return
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.trading_halted = True
            self.halt_reason = f"Üst üste {self.consecutive_losses} kayıp yaşandı."

    def reset_halt(self) -> None:
        self.trading_halted = False
        self.halt_reason = None
        self.consecutive_losses = 0
        self._save()

    def reset_portfolio(self) -> None:
        self.equity = self.config.initial_balance
        self.peak_equity = self.config.initial_balance
        self.day_start_equity = self.config.initial_balance
        self.day_date = date.today().isoformat()
        self.open_positions = {}
        self.trade_history = []
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = None
        self._save()

    # ------------------------------------------------------------ sizing
    def _calc_stop_loss(self, entry: float, atr: float, direction: Direction) -> float:
        dist = atr * self.config.stop_loss_atr_multiplier
        return entry - dist if direction == Direction.LONG else entry + dist

    def _calc_take_profit(self, entry: float, stop: float, direction: Direction) -> float:
        risk_dist = abs(entry - stop)
        reward_dist = risk_dist * self.config.min_rr
        return entry + reward_dist if direction == Direction.LONG else entry - reward_dist

    def _calc_liquidation_price(self, entry: float, leverage: float, direction: Direction) -> Optional[float]:
        """
        İzole marjda basitleştirilmiş likidasyon fiyatı: leverage=L iken fiyat
        entry'nin yaklaşık 1/L'i kadar aleyhe hareket ederse marj tükenir
        (fonlama/bakım marjı hariç, kaba bir yaklaşımdır — gerçek borsalar
        bakım marjı yüzünden biraz daha erken likide eder).

        leverage <= 1 ise pozisyon zaten fazla teminatlıdır (sermayenin
        tamamından azı kullanılıyor) — gerçek bir likidasyon riski yoktur,
        bu yüzden None döner. (leverage=1 formülü entry-entry=0 verir ki bu
        LONG için anlamsızdır; leverage<1'de ise formül entry'den daha
        büyük bir "hareket" hesaplayıp negatif/anlamsız bir fiyat üretir.)
        """
        if leverage <= 1.0:
            return None
        move = entry / leverage
        return entry - move if direction == Direction.LONG else entry + move

    def _calc_position_size(
        self, entry: float, stop: float, direction: Direction
    ) -> tuple[float, float, Optional[float]]:
        risk_dist = abs(entry - stop)
        if risk_dist <= 0:
            return 0.0, 0.0, None

        risk_amount = self.equity * self.config.risk_per_trade_pct
        size = risk_amount / risk_dist
        notional = size * entry

        max_notional = self.equity * self.config.max_leverage
        if notional > max_notional:
            size = max_notional / entry
            notional = max_notional

        leverage = notional / self.equity if self.equity > 0 else 0.0

        # Likidasyon güvenliği (yalnızca leverage > 1x'te anlamlı — 1x ve altı
        # zaten fazla teminatlı, likidasyon riski yok). stop-loss, likidasyon
        # tetiklenmeden ÖNCE gerçekleşmeli; değilse kaldıracı, stop mesafesi
        # güvenli tampon içinde kalacak şekilde aşağı çek (pozisyonu tamamen
        # reddetmek yerine — bu genelde çok düşük volatilitede/çok sıkı
        # stop'larda devreye girer).
        if leverage > 1.0:
            liq_distance = (entry / leverage) * self.config.liquidation_safety_buffer
            if risk_dist > liq_distance:
                safe_leverage = (entry / risk_dist) * self.config.liquidation_safety_buffer
                safe_leverage = min(safe_leverage, self.config.max_leverage)
                notional = self.equity * safe_leverage
                size = notional / entry if entry > 0 else 0.0
                leverage = safe_leverage

        liquidation_price = self._calc_liquidation_price(entry, leverage, direction)
        return size, leverage, liquidation_price

    # ------------------------------------------------------------ trading
    def evaluate_trade(self, symbol: str, direction: Direction, entry_price: float, atr: float) -> TradeDecision:
        self._roll_day()
        self._update_breakers()
        if self.trading_halted:
            return TradeDecision(False, f"Trading durduruldu: {self.halt_reason}")
        if not (entry_price > 0) or not (atr > 0):
            return TradeDecision(False, "Geçersiz fiyat veya ATR.")
        if len(self.open_positions) >= self.config.max_open_positions:
            return TradeDecision(
                False, f"Maksimum açık pozisyon sayısına ulaşıldı ({self.config.max_open_positions})."
            )
        if symbol in self.open_positions:
            return TradeDecision(False, f"'{symbol}' için zaten açık pozisyon var.")

        stop = self._calc_stop_loss(entry_price, atr, direction)
        take = self._calc_take_profit(entry_price, stop, direction)
        size, leverage, liquidation_price = self._calc_position_size(entry_price, stop, direction)
        if not (size > 0):
            return TradeDecision(False, "Hesaplanan pozisyon büyüklüğü sıfır.")
        if leverage > self.config.max_leverage + 1e-9:
            return TradeDecision(False, f"Gerekli kaldıraç ({leverage:.2f}x) limiti aşıyor.")

        return TradeDecision(
            True, "Onaylandı.", size=size, stop_loss=stop, take_profit=take,
            leverage=leverage, liquidation_price=liquidation_price,
        )

    def open_position(self, symbol: str, direction: Direction, entry_price: float, decision: TradeDecision) -> None:
        self.open_positions[symbol] = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=decision.size,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            leverage=decision.leverage,
            liquidation_price=decision.liquidation_price,
        )
        self._save()

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[ClosedTrade]:
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None
        dir_sign = 1 if pos.direction == Direction.LONG else -1
        pnl = (exit_price - pos.entry_price) * pos.size * dir_sign
        notional = pos.size * pos.entry_price
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
        trade = ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=(pnl / notional if notional > 0 else 0.0),
            reason=reason,
        )
        self.trade_history.append(trade)
        self._update_breakers()
        return trade

    def check_exits(self, current_prices: dict[str, float]) -> list[ClosedTrade]:
        """
        Açık her pozisyonu KENDİ sembolünün fiyatıyla kontrol eder.
        `current_prices`, birden fazla sembolün son bilinen fiyatını
        içerebilir (ör. {"BTCUSDT": 65000, "ETHUSDT": 3400}) — böylece
        aynı anda birden fazla sembolde açık pozisyon varken biri
        diğerinin fiyatıyla yanlışlıkla kapatılmaz.
        """
        closed: list[ClosedTrade] = []
        for symbol in list(self.open_positions.keys()):
            price = current_prices.get(symbol)
            if price is None:
                continue
            pos = self.open_positions[symbol]
            hit_liquidation = pos.liquidation_price is not None and (
                price <= pos.liquidation_price if pos.direction == Direction.LONG else price >= pos.liquidation_price
            )
            hit_stop = price <= pos.stop_loss if pos.direction == Direction.LONG else price >= pos.stop_loss
            hit_target = price >= pos.take_profit if pos.direction == Direction.LONG else price <= pos.take_profit
            trade = None
            if hit_liquidation:
                # Ani fiyat sıçraması stop-loss'u atlayıp doğrudan likidasyona
                # ulaştıysa (gap), gerçek borsa davranışını yansıtmak için
                # likidasyon fiyatından kapat — stop fiyatından değil.
                trade = self._close_position(symbol, pos.liquidation_price, "liquidation")
            elif hit_stop:
                trade = self._close_position(symbol, price, "stop_loss")
            elif hit_target:
                trade = self._close_position(symbol, price, "take_profit")
            if trade:
                closed.append(trade)
        if closed:
            self._save()
        return closed

    def portfolio_summary(self) -> dict:
        self._roll_day()
        self._update_breakers()
        return {
            "equity": self.equity,
            "peakEquity": self.peak_equity,
            "drawdownPct": self.drawdown_pct(),
            "dailyPnlPct": self.daily_pnl_pct(),
            "openPositions": len(self.open_positions),
            "consecutiveLosses": self.consecutive_losses,
            "tradingHalted": self.trading_halted,
            "haltReason": self.halt_reason,
        }
