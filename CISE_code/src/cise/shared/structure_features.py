from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


STRUCTURE_FEATURES = [
    "number_of_sites",
    "number_of_unique_elements",
    "density_g_cm3",
    "volume_angstrom3",
    "volume_per_atom_angstrom3",
    "lattice_a_angstrom",
    "lattice_b_angstrom",
    "lattice_c_angstrom",
    "lattice_alpha_degree",
    "lattice_beta_degree",
    "lattice_gamma_degree",
    "minimum_interatomic_distance_angstrom",
    "atomic_number_mean",
    "atomic_number_std",
    "atomic_number_min",
    "atomic_number_max",
    "electronegativity_mean",
    "electronegativity_std",
    "electronegativity_min",
    "electronegativity_max",
    "atomic_radius_angstrom_mean",
    "atomic_radius_angstrom_std",
    "atomic_radius_angstrom_min",
    "atomic_radius_angstrom_max",
    "structure_parse_success_flag",
    "structure_validity_flag",
    "composition_property_missing_flag",
]


def weighted_stats(values: list[float], weights: list[float]) -> tuple[float, float, float, float]:
    finite = [(value, weight) for value, weight in zip(values, weights) if math.isfinite(value) and weight > 0]
    if not finite:
        return (math.nan,) * 4
    array = np.asarray([item[0] for item in finite], dtype=float)
    mass = np.asarray([item[1] for item in finite], dtype=float)
    mass /= mass.sum()
    mean = float(np.dot(array, mass))
    variance = float(np.dot((array - mean) ** 2, mass))
    return mean, math.sqrt(max(0.0, variance)), float(array.min()), float(array.max())


def composition_values(structure: Any) -> tuple[list[Any], list[float]]:
    from pymatgen.core import Element

    elements = []
    weights = []
    for element, amount in structure.composition.element_composition.items():
        elements.append(Element(str(element)))
        weights.append(float(amount))
    return elements, weights


def structure_feature_values(cif_path: Path) -> dict[str, float]:
    from pymatgen.core import Structure

    missing = {name: math.nan for name in STRUCTURE_FEATURES}
    missing["structure_parse_success_flag"] = 0.0
    missing["structure_validity_flag"] = 0.0
    missing["composition_property_missing_flag"] = 1.0
    try:
        structure = Structure.from_file(cif_path)
        elements, weights = composition_values(structure)
        atomic_numbers = [float(element.Z) for element in elements]
        electronegativities = [float(element.X) if element.X is not None else math.nan for element in elements]
        radii = []
        for element in elements:
            radius = element.atomic_radius
            if radius is None:
                radius = element.atomic_radius_calculated
            radii.append(float(radius) if radius is not None else math.nan)
        z_stats = weighted_stats(atomic_numbers, weights)
        x_stats = weighted_stats(electronegativities, weights)
        radius_stats = weighted_stats(radii, weights)
        distances = np.asarray(structure.distance_matrix, dtype=float)
        if len(structure) > 1:
            distances[np.diag_indices_from(distances)] = np.inf
            minimum_distance = float(np.min(distances))
        else:
            minimum_distance = math.nan
        volume = float(structure.volume)
        validity = bool(
            len(structure) > 0
            and math.isfinite(volume)
            and volume > 0
            and all(math.isfinite(value) and value > 0 for value in structure.lattice.abc)
        )
        a, b, c = (float(value) for value in structure.lattice.abc)
        alpha, beta, gamma = (float(value) for value in structure.lattice.angles)
        return {
            "number_of_sites": float(len(structure)),
            "number_of_unique_elements": float(len(elements)),
            "density_g_cm3": float(structure.density),
            "volume_angstrom3": volume,
            "volume_per_atom_angstrom3": volume / len(structure),
            "lattice_a_angstrom": a,
            "lattice_b_angstrom": b,
            "lattice_c_angstrom": c,
            "lattice_alpha_degree": alpha,
            "lattice_beta_degree": beta,
            "lattice_gamma_degree": gamma,
            "minimum_interatomic_distance_angstrom": minimum_distance,
            "atomic_number_mean": z_stats[0],
            "atomic_number_std": z_stats[1],
            "atomic_number_min": z_stats[2],
            "atomic_number_max": z_stats[3],
            "electronegativity_mean": x_stats[0],
            "electronegativity_std": x_stats[1],
            "electronegativity_min": x_stats[2],
            "electronegativity_max": x_stats[3],
            "atomic_radius_angstrom_mean": radius_stats[0],
            "atomic_radius_angstrom_std": radius_stats[1],
            "atomic_radius_angstrom_min": radius_stats[2],
            "atomic_radius_angstrom_max": radius_stats[3],
            "structure_parse_success_flag": 1.0,
            "structure_validity_flag": float(validity),
            "composition_property_missing_flag": float(
                any(not math.isfinite(value) for value in electronegativities + radii)
            ),
        }
    except Exception:
        return missing

