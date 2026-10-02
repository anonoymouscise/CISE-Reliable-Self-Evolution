"""Quantum ESPRESSO output parsing used by the validation worker."""

import re

RY_TO_EV = 13.605693122994


FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def parse_total_energy(text: str, require_converged_marker: bool = True) -> float | None:
    prefix = r"!\s+" if require_converged_marker else r"\s*"
    matches = re.findall(prefix + r"total energy\s+=\s+(" + FLOAT_PATTERN.pattern + r")\s+Ry", text)
    return float(matches[-1]) * RY_TO_EV if matches else None


def parse_scf_converged(text: str) -> bool:
    if re.search(r"convergence\s+NOT\s+achieved", text, re.IGNORECASE):
        return False
    return bool(parse_total_energy(text, require_converged_marker=True))


def parse_fermi(text: str) -> float | None:
    matches = re.findall(r"the Fermi energy is\s+([-+]?\d+(?:\.\d+)?)\s+ev", text, re.IGNORECASE)
    return float(matches[-1]) if matches else None


def parse_pw_version(text: str) -> str | None:
    match = re.search(r"Program\s+(PWSCF\s+v\.[^\s]+)", text)
    return match.group(1) if match else None
