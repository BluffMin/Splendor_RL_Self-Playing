"""Policy-facing canonical payment adapter over the exact rules engine."""

from __future__ import annotations

from typing import Literal

from ..actions import PAYMENT_OFFSET
from ..core import EngineStep, Phase, SplendorGame


class CanonicalPaymentWrapper:
    def __init__(
        self,
        game: SplendorGame,
        payment_mode: Literal["canonical", "exact"] = "canonical",
    ) -> None:
        if payment_mode not in {"canonical", "exact"}:
            raise ValueError("payment_mode must be canonical or exact")
        self.game = game
        self.payment_mode = payment_mode
        self.canonical_payments = 0
        game.add_event_listener(self._label_canonical_event)

    def _label_canonical_event(self, event: dict) -> None:
        if (
            getattr(self, "_selecting_canonical", False)
            and event.get("phase_before") == "payment"
        ):
            payment = event.get("action_params", {})
            colored = payment.get("colored", {})
            event.update(
                {
                    "action_type": "canonical_payment",
                    "action_text": "canonical payment",
                    "automatic": True,
                    "selected_plan_index": 0,
                    "payment": {**colored, "gold": payment.get("total_gold", 0)},
                }
            )

    def policy_step(self, action: int) -> tuple[EngineStep, tuple[EngineStep, ...]]:
        result = self.game.step(action)
        automatic: list[EngineStep] = []
        if self.payment_mode == "canonical" and self.game.phase == Phase.PAYMENT:
            # The exact engine orders plans by gold usage, then deterministically.
            self._selecting_canonical = True
            try:
                automatic.append(self.game.step(PAYMENT_OFFSET))
                self.canonical_payments += 1
            finally:
                self._selecting_canonical = False
        return result, tuple(automatic)

    def legal_actions(self) -> list[int]:
        if self.payment_mode == "canonical" and self.game.phase == Phase.PAYMENT:
            raise RuntimeError("PAYMENT is automatic in canonical mode")
        return self.game.legal_actions()
