"""Read suggested plane-wave cutoffs from UPF pseudopotentials."""

import re
from pathlib import Path

FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def pseudo_cutoffs(path: str) -> tuple[float | None, float | None]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    wfc_patterns = [
        r"Suggested minimum cutoff for wavefunctions:\s*(" + FLOAT_RE.pattern + r")",
        r"wfc_cutoff\s*=\s*['\"]?(" + FLOAT_RE.pattern + r")",
    ]
    rho_patterns = [
        r"Suggested minimum cutoff for charge density:\s*(" + FLOAT_RE.pattern + r")",
        r"rho_cutoff\s*=\s*['\"]?(" + FLOAT_RE.pattern + r")",
    ]

    def first_positive(patterns: list[str]) -> float | None:
        for pat in patterns:
            match = re.search(pat, text, flags=re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if value > 0:
                    return value
        return None

    return first_positive(wfc_patterns), first_positive(rho_patterns)
