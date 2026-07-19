from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal
import pandas as pd

Side = Literal["buy", "sell"]
Bias = Literal["bullish", "bearish", "neutral"]

@dataclass(frozen=True)
class TradeEvent:
    event_time: datetime
    exchange: str
    symbol: str
    price: float
    quantity: float
    taker_side: Side
    trade_id: str | None = None
    receive_time: datetime | None = None

    @property
    def notional(self) -> float:
        return self.price * self.quantity

@dataclass
class Zone:
    zone_id: str
    kind: str
    side: Literal["above", "below"]
    low: float
    high: float
    score: float
    created_at: pd.Timestamp | datetime
    available_at: pd.Timestamp | datetime
    expires_at: pd.Timestamp | datetime | None = None
    swept_at: pd.Timestamp | datetime | None = None
    touches: int = 0
    sources: tuple[str, ...] = ()

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class PredictorOutput:
    timestamp: datetime
    bias: Bias
    setup_type: str | None = None
    zone: str | None = None
    sweep_status: str = "none"
    orderflow_confirmation: bool = False
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reward_risk: float | None = None
    probability_tp_before_sl: float | None = None
    position_size: float = 0.0
    no_trade_reason: str | None = None
