from dataclasses import dataclass, field
from typing import List


@dataclass
class GateCheck:
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def reason_str(self):
        return "; ".join(self.reasons) if self.reasons else ""


def combine_gate_checks(*checks):
    ok = True
    reasons = []
    for check in checks:
        if check is None:
            continue
        if not check.ok:
            ok = False
            reasons.extend(check.reasons)
    return GateCheck(ok=ok, reasons=reasons)
