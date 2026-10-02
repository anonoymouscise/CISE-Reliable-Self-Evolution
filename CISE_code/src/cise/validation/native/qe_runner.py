from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element, Structure

from .qe_output import (
    RY_TO_EV,
    parse_fermi,
    parse_pw_version,
    parse_scf_converged,
    parse_total_energy,
)
from .pseudopotentials import pseudo_cutoffs


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]


def find_pseudopotentials_exact(elements: list[str], pseudo_dir: str) -> dict[str, str]:

    root = Path(pseudo_dir).resolve()
    files = list(root.rglob("*.UPF")) + list(root.rglob("*.upf"))
    mapping = {}
    for element in elements:
        pattern = re.compile(rf"(^|[/_.-]){re.escape(element)}([_.-]|$)", re.IGNORECASE)
        matches = [path for path in files if pattern.search(path.name)]
        if not matches:
            matches = [path for path in files if path.name.lower().startswith(element.lower() + ".")]
        if not matches:
            raise FileNotFoundError(f"No UPF pseudopotential found for {element} in {root}")
        mapping[element] = str(sorted(matches, key=lambda path: (len(path.name), path.name))[0])
    return mapping


def length_mesh(structure: Structure, length: float) -> tuple[int, int, int]:

    reciprocal = np.linalg.inv(np.asarray(structure.lattice.matrix, dtype=float)).T
    return tuple(max(1, int(length * float(np.linalg.norm(vec)) + 0.5)) for vec in reciprocal)


def upf_valence(path: str) -> float:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"z_valence\s*=\s*['\"]?\s*(" + FLOAT_RE.pattern + r")",
        r"Z valence\s*=\s*(" + FLOAT_RE.pattern + r")",
        r"(" + FLOAT_RE.pattern + r")\s+Z valence",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError(f"Could not parse z_valence from {path}")


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def magnetic_initialization(
    structure: Structure, row: dict[str, str], pseudo_map: dict[str, str]
) -> dict[str, Any]:

    source = finite_float(row.get("jarvis_magmom_outcar_muB_cell"))
    source_field = "jarvis_magmom_outcar_muB_cell"
    if source is None:
        source = finite_float(row.get("jarvis_magmom_oszicar_muB_cell"))
        source_field = "jarvis_magmom_oszicar_muB_cell"
    magnetic_indices = [
        index
        for index, site in enumerate(structure)
        if Element(str(site.specie)).block in {"d", "f"}
    ]
    source_magnetic = source is not None and abs(source) > 0.1
    spin_polarized = source_magnetic or bool(magnetic_indices)
    if not spin_polarized:
        return {
            "spin_polarized": False,
            "source_moment_muB_cell": source,
            "source_moment_field": source_field if source is not None else None,
            "initial_moment_muB_cell": 0.0,
            "initial_atomic_moments_muB": [0.0] * len(structure),
            "qe_starting_magnetization": {},
            "initialization": "spin-paired: no source moment and no d/f-block atom",
        }
    if not magnetic_indices:
        magnetic_indices = list(range(len(structure)))
    target = abs(source) if source_magnetic else float(len(magnetic_indices))
    per_site = min(5.0, max(0.5, target / len(magnetic_indices)))
    atomic = [per_site if index in magnetic_indices else 0.0 for index in range(len(structure))]
    by_species: dict[str, list[float]] = {}
    for site, moment in zip(structure, atomic):
        by_species.setdefault(str(site.specie), []).append(moment)
    qe_seed = {
        element: min(0.8, max(0.0, sum(values) / len(values) / upf_valence(pseudo_map[element])))
        for element, values in by_species.items()
        if any(value > 0 for value in values)
    }
    return {
        "spin_polarized": True,
        "source_moment_muB_cell": source,
        "source_moment_field": source_field if source is not None else None,
        "initial_moment_muB_cell": float(sum(atomic)),
        "initial_atomic_moments_muB": atomic,
        "qe_starting_magnetization": qe_seed,
        "initialization": "collinear ferromagnetic seed; final total moment unconstrained",
        "limitation": "source has no site-resolved magnetic order; AFM/ferrimagnetic states are not enumerated",
    }


def protocol_parameters(
    structure: Structure,
    row: dict[str, str],
    pseudo_map: dict[str, str],
    scf_length: float,
    band_length: float,
    ecutwfc_override: float | None = None,
    ecutrho_override: float | None = None,
    scf_kgrid_override: tuple[int, int, int] | None = None,
    band_kgrid_override: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    cutoffs = {element: pseudo_cutoffs(path) for element, path in pseudo_map.items()}
    wfc = [value[0] for value in cutoffs.values() if value[0] is not None]
    rho = [value[1] for value in cutoffs.values() if value[1] is not None]
    ecutwfc = max([60.0, *wfc]) if wfc else 60.0


    ecutrho = max([8.0 * ecutwfc, *rho]) if rho else 8.0 * ecutwfc
    cutoff_policy = "pseudopotential_suggested_with_conservative_floor"
    if ecutwfc_override is not None:
        ecutwfc = float(ecutwfc_override)
        cutoff_policy = "explicit_resolution_sensitivity_override"
    if ecutrho_override is not None:
        ecutrho = float(ecutrho_override)
        cutoff_policy = "explicit_resolution_sensitivity_override"
    counts: dict[str, int] = {}
    for site in structure:
        element = str(site.specie)
        counts[element] = counts.get(element, 0) + 1
    electrons = sum(counts[element] * upf_valence(pseudo_map[element]) for element in counts)
    magnetism = magnetic_initialization(structure, row, pseudo_map)
    if magnetism["spin_polarized"]:
        occupied_bands = int(
            math.ceil((electrons + float(magnetism["initial_moment_muB_cell"])) / 2.0)
        )
    else:
        occupied_bands = int(math.ceil(electrons / 2.0))
    nbands = occupied_bands + max(8, int(math.ceil(0.2 * occupied_bands)))
    return {
        "pseudo_cutoffs_Ry": {
            element: {"suggested_wfc": values[0], "suggested_rho": values[1]}
            for element, values in cutoffs.items()
        },
        "ecutwfc_Ry": float(ecutwfc),
        "ecutrho_Ry": float(ecutrho),
        "cutoff_policy": cutoff_policy,
        "valence_electrons": float(electrons),
        "maximum_initially_occupied_bands_per_spin": occupied_bands,
        "nbands": nbands,
        "scf_kgrid": scf_kgrid_override or length_mesh(structure, scf_length),
        "band_kgrid": band_kgrid_override or length_mesh(structure, band_length),
        "kgrid_policy": (
            "explicit_resolution_sensitivity_override"
            if scf_kgrid_override is not None or band_kgrid_override is not None
            else "reciprocal_length"
        ),
        "scf_length_A": scf_length,
        "band_length_A": band_length,
        "occupations": "Fermi-Dirac 0.05 eV",
        "magnetism": magnetism,
        "fixed_structure": True,
        "xc": "PBE",
        "soc": False,
        "dft_u": False,
    }


def run_to_file(
    command: list[str],
    cwd: Path,
    output: Path,
    timeout: int | None,
    env: dict[str, str],
) -> dict:
    started = time.time()
    try:
        with output.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        return {
            "returncode": proc.returncode,
            "stderr": proc.stderr[-8000:],
            "timeout": False,
            "wall_seconds": time.time() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stderr": (exc.stderr or "")[-8000:] + f"\nTimed out after {timeout} seconds",
            "timeout": True,
            "wall_seconds": time.time() - started,
        }


def qe_input(
    structure: Structure,
    pseudo_map: dict[str, str],
    prefix: str,
    outdir: Path,
    calculation: str,
    kgrid: tuple[int, int, int],
    ecutwfc: float,
    ecutrho: float,
    nbands: int | None,
    robust_electrons: int,
    magnetism: dict[str, Any],
    conv_thr_override: float | None = None,
) -> str:
    species = sorted({str(site.specie) for site in structure})
    pseudo_directories = {str(Path(path).resolve().parent) for path in pseudo_map.values()}
    if len(pseudo_directories) != 1:
        raise ValueError(f"QE requires one pinned pseudopotential directory: {pseudo_directories}")
    pseudo_directory = next(iter(pseudo_directories))
    lines = [
        "&CONTROL",
        f"  calculation = '{calculation}',",
        f"  prefix = '{prefix}',",
        f"  outdir = '{outdir}',",


        f"  pseudo_dir = '{pseudo_directory}',",
        "  verbosity = 'high',",
        "/",
        "&SYSTEM",
        "  ibrav = 0,",
        f"  nat = {len(structure)},",
        f"  ntyp = {len(species)},",
        f"  ecutwfc = {ecutwfc:.8g},",
        f"  ecutrho = {ecutrho:.8g},",
        "  occupations = 'smearing',",
        "  smearing = 'fd',",
        f"  degauss = {0.05 / RY_TO_EV:.10g},",
    ]
    if nbands is not None:
        lines.append(f"  nbnd = {nbands},")
    if magnetism["spin_polarized"]:
        lines.append("  nspin = 2,")
        for index, element in enumerate(species, 1):
            value = magnetism["qe_starting_magnetization"].get(element)
            if value is not None:
                lines.append(f"  starting_magnetization({index}) = {value:.8g},")
    lines.extend(["/", "&ELECTRONS"])
    if conv_thr_override is not None:
        lines.append(f"  conv_thr = {conv_thr_override:.8g},")
    if robust_electrons >= 1:
        lines.extend(["  electron_maxstep = 300,", "  mixing_beta = 0.3,"])
    if robust_electrons >= 2:
        lines[-2:] = [
            "  electron_maxstep = 600,",
            "  mixing_beta = 0.1,",
            "  mixing_mode = 'local-TF',",
        ]
    lines.extend(["/", "ATOMIC_SPECIES"])
    for element in species:
        lines.append(
            f"  {element} {float(Element(element).atomic_mass):.8f} {Path(pseudo_map[element]).name}"
        )
    lines.append("CELL_PARAMETERS angstrom")
    for vector in structure.lattice.matrix:
        lines.append("  " + " ".join(f"{value:.12f}" for value in vector))
    lines.append("ATOMIC_POSITIONS crystal")
    for site in structure:
        lines.append(f"  {site.specie} " + " ".join(f"{value:.12f}" for value in site.frac_coords))
    lines.extend(["K_POINTS automatic", f"  {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0"])
    return "\n".join(lines) + "\n"


def parse_qe_bandgap(text: str, fermi: float | None) -> dict[str, Any]:
    blocks: list[dict[str, list[float]]] = []
    current = None
    mode = None
    for raw in text.splitlines():
        line = raw.strip()
        lower = line.lower()
        if "bands (ev)" in lower:
            current = {"eigenvalues": [], "occupations": []}
            blocks.append(current)
            mode = "eigenvalues"
            continue
        if "occupation numbers" in lower and current is not None:
            mode = "occupations"
            continue
        if current is None or mode is None:
            continue
        numbers = [float(match.group(0)) for match in FLOAT_RE.finditer(line)]
        if numbers:
            current[mode].extend(numbers)
        elif current[mode]:
            mode = None

    occupied: list[float] = []
    empty: list[float] = []
    partial = False
    valid_blocks = 0
    for block in blocks:
        eigs, occs = block["eigenvalues"], block["occupations"]
        if len(eigs) != len(occs) or not eigs:
            continue
        valid_blocks += 1
        for eigenvalue, occupation in zip(eigs, occs):


            if 1.0e-3 < occupation < 0.999:
                partial = True
            if occupation >= 0.5:
                occupied.append(eigenvalue)
            else:
                empty.append(eigenvalue)
    if partial:
        return {"band_gap_eV": 0.0, "method": "partial_occupation_metal", "fermi_eV": fermi,
                "valid_kpoint_blocks": valid_blocks}
    if occupied and empty:
        return {"band_gap_eV": max(0.0, min(empty) - max(occupied)), "method": "occupation_split",
                "fermi_eV": fermi, "valid_kpoint_blocks": valid_blocks}
    all_eigs = [value for block in blocks for value in block["eigenvalues"]]
    if fermi is not None and all_eigs:
        below = [value for value in all_eigs if value <= fermi]
        above = [value for value in all_eigs if value > fermi]
        if below and above:
            return {"band_gap_eV": max(0.0, min(above) - max(below)), "method": "fermi_split",
                    "fermi_eV": fermi, "valid_kpoint_blocks": valid_blocks}
    return {"band_gap_eV": None, "method": "parse_failed", "fermi_eV": fermi,
            "valid_kpoint_blocks": valid_blocks}


def base_record(args: argparse.Namespace, row: dict[str, str], structure: Structure, params: dict) -> dict:
    cif = Path(row[args.manifest_cif_column]).resolve()
    return {
        "created_at": datetime.now().isoformat(),
        "status": "running",
        "code": args.code,
        "label": row.get("label") or row.get("jid") or cif.stem,
        "jid": row.get("jid"),
        "input_cif": str(cif),
        "cif_sha256": sha256_file(cif),
        "formula": structure.composition.reduced_formula,
        "num_sites": len(structure),
        "parameters": params,
        "interpretation": "Fixed-structure Quantum ESPRESSO validation; protocol is recorded in parameters.",
    }


def run_qe(args: argparse.Namespace, row: dict[str, str], run_dir: Path) -> dict:
    cif = Path(row[args.manifest_cif_column]).resolve()
    structure = Structure.from_file(cif, primitive=False)
    elements = sorted({str(site.specie) for site in structure})
    pseudo_map = find_pseudopotentials_exact(elements, args.pseudo_dir)
    params = protocol_parameters(
        structure,
        row,
        pseudo_map,
        args.scf_length,
        args.band_length,
        args.ecutwfc_override,
        args.ecutrho_override,
        tuple(args.scf_kgrid_override) if args.scf_kgrid_override is not None else None,
        tuple(args.band_kgrid_override) if args.band_kgrid_override is not None else None,
    )
    if args.conv_thr_override is not None:
        params["conv_thr_Ry"] = float(args.conv_thr_override)
    params["nscf_mpi_processes"] = int(args.qe_nscf_mpi_processes)
    record = base_record(args, row, structure, params)
    record["pseudopotentials"] = pseudo_map
    record["pseudopotential_sha256"] = {
        element: sha256_file(Path(path)) for element, path in pseudo_map.items()
    }
    for path in pseudo_map.values():
        shutil.copy2(path, run_dir / Path(path).name)

    pw = shutil.which(os.environ.get("CISE_PW", "pw.x")) or "pw.x"
    version = subprocess.run(
        [pw, "-h"], capture_output=True, text=True,
        timeout=args.qe_version_timeout, check=False,
    )
    record["pw_x"] = pw
    record["pw_x_version"] = parse_pw_version(version.stdout)
    prefix = safe_label(str(record["label"]))
    qe_tmp = run_dir / "qe_tmp"
    qe_tmp.mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(args.omp_threads)}

    scf_path = run_dir / "scf.in"
    scf_path.write_text(
        qe_input(structure, pseudo_map, prefix, qe_tmp, "scf", params["scf_kgrid"],
                 params["ecutwfc_Ry"], params["ecutrho_Ry"], None, False,
                 params["magnetism"], args.conv_thr_override),
        encoding="utf-8",
    )
    scf_run = run_to_file([pw, "-in", "scf.in"], run_dir, run_dir / "scf.out", args.timeout, env)
    scf_text = (run_dir / "scf.out").read_text(encoding="utf-8", errors="replace")
    retry_run = None
    retry2_run = None
    if (
        not parse_scf_converged(scf_text)
        and not scf_run["timeout"]
        and not args.disable_scf_retries
    ):
        retry_path = run_dir / "scf_retry.in"
        retry_path.write_text(
            qe_input(structure, pseudo_map, prefix, qe_tmp, "scf", params["scf_kgrid"],
                     params["ecutwfc_Ry"], params["ecutrho_Ry"], None, True,
                     params["magnetism"], args.conv_thr_override),
            encoding="utf-8",
        )
        retry_run = run_to_file([pw, "-in", "scf_retry.in"], run_dir, run_dir / "scf_retry.out",
                                args.timeout, env)
        retry_text = (run_dir / "scf_retry.out").read_text(encoding="utf-8", errors="replace")
        if parse_scf_converged(retry_text):
            scf_text = retry_text
        elif not retry_run["timeout"]:
            retry2_path = run_dir / "scf_retry2.in"
            retry2_path.write_text(
                qe_input(structure, pseudo_map, prefix, qe_tmp, "scf", params["scf_kgrid"],
                         params["ecutwfc_Ry"], params["ecutrho_Ry"], None, 2,
                         params["magnetism"], args.conv_thr_override),
                encoding="utf-8",
            )
            retry2_run = run_to_file(
                [pw, "-in", "scf_retry2.in"], run_dir, run_dir / "scf_retry2.out",
                args.timeout, env,
            )
            retry2_text = (run_dir / "scf_retry2.out").read_text(
                encoding="utf-8", errors="replace"
            )
            if parse_scf_converged(retry2_text):
                scf_text = retry2_text

    scf_converged = parse_scf_converged(scf_text)
    nscf_run = None
    band_gap = None
    if scf_converged and not args.skip_bands:


        restart_dir = qe_tmp / f"{prefix}.save"
        if not restart_dir.is_dir():
            raise FileNotFoundError(f"QE SCF restart directory is missing: {restart_dir}")
        for path in pseudo_map.values():
            shutil.copy2(path, restart_dir / Path(path).name)
        nscf_path = run_dir / "nscf.in"
        nscf_path.write_text(
            qe_input(structure, pseudo_map, prefix, qe_tmp, "nscf", params["band_kgrid"],
                     params["ecutwfc_Ry"], params["ecutrho_Ry"], params["nbands"], True,
                     params["magnetism"], args.conv_thr_override),
            encoding="utf-8",
        )
        nscf_command = [pw, "-in", "nscf.in"]
        nscf_env = env
        if args.qe_nscf_mpi_processes > 1:
            mpirun = shutil.which("mpirun") or str(Path(pw).with_name("mpirun"))
            if not Path(mpirun).is_file():
                raise RuntimeError(f"QE NSCF MPI launcher does not exist: {mpirun}")
            nscf_command = [
                mpirun,
                "--bind-to", "core",
                "-np", str(args.qe_nscf_mpi_processes),
                pw,
                "-nk", str(args.qe_nscf_mpi_processes),
                "-in", "nscf.in",
            ]
            nscf_env = {**env, "OMP_NUM_THREADS": "1"}
        nscf_run = run_to_file(
            nscf_command, run_dir, run_dir / "nscf.out", args.timeout, nscf_env
        )
        nscf_text = (run_dir / "nscf.out").read_text(encoding="utf-8", errors="replace")
        band_gap = parse_qe_bandgap(nscf_text, parse_fermi(nscf_text or scf_text))

    gap_resolved = args.skip_bands or (
        band_gap is not None and band_gap.get("band_gap_eV") is not None
    )
    record.update(
        {
            "status": "completed" if scf_converged and gap_resolved and (
                args.skip_bands or (nscf_run and nscf_run["returncode"] == 0)
            ) else "failed",
            "scf_converged": scf_converged,
            "scf_run": scf_run,
            "scf_retry_run": retry_run,
            "scf_retry2_run": retry2_run,
            "nscf_run": nscf_run,
            "total_energy_eV_cell": parse_total_energy(scf_text),
            "energy_eV_atom": (parse_total_energy(scf_text) / len(structure)) if parse_total_energy(scf_text) is not None else None,
            "band_gap": band_gap,
        }
    )
    return record


def single(args: argparse.Namespace) -> int:
    row = json.loads(args.row_json)
    index = int(args.row_index)
    label = safe_label(row.get("label") or row.get("jid") or f"row_{index:04d}")
    run_dir = Path(args.output_dir).resolve() / "profiles" / f"{index:04d}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        record = run_qe(args, row, run_dir)
    except Exception as exc:
        record = {
            "created_at": datetime.now().isoformat(),
            "status": "failed",
            "code": args.code,
            "label": label,
            "jid": row.get("jid"),
            "input_cif": row.get(args.manifest_cif_column),
            "reason": repr(exc),
        }
    summary_path = run_dir / "summary.json"


    write_json(summary_path, record)
    if args.code == "qe" and not args.keep_qe_tmp:
        qe_tmp = run_dir / "qe_tmp"
        try:
            if qe_tmp.is_dir():
                shutil.rmtree(qe_tmp)
            record["qe_tmp_retained"] = False
        except OSError as exc:
            record["qe_tmp_retained"] = True
            record["qe_tmp_cleanup_error"] = repr(exc)
        write_json(summary_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record.get("status") == "completed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--code", choices=["qe"], required=True)
    parser.add_argument("--manifest-cif-column", default="audit_cif")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-bands", action="store_true")
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--row-json")
    parser.add_argument(
        "--pseudo-dir",
        default=os.environ.get("ESPRESSO_PSEUDO"),
    )
    parser.add_argument("--scf-length", type=float, default=20.0)
    parser.add_argument("--band-length", type=float, default=30.0)
    parser.add_argument(
        "--ecutwfc-override",
        type=float,
        help="explicit low-resolution sensitivity cutoff in Ry; bypasses suggested-cutoff floors",
    )
    parser.add_argument(
        "--ecutrho-override",
        type=float,
        help="explicit low-resolution sensitivity charge-density cutoff in Ry",
    )
    parser.add_argument(
        "--conv-thr-override",
        type=float,
        help="explicit QE electronic convergence threshold in Ry for sensitivity benchmarks",
    )
    parser.add_argument(
        "--disable-scf-retries",
        action="store_true",
        help="do not spend additional wall time on robust SCF retries",
    )
    parser.add_argument(
        "--scf-kgrid-override",
        type=int,
        nargs=3,
        metavar=("KX", "KY", "KZ"),
        help="explicit SCF k-grid for resolution-sensitivity benchmarks",
    )
    parser.add_argument(
        "--band-kgrid-override",
        type=int,
        nargs=3,
        metavar=("KX", "KY", "KZ"),
        help="explicit NSCF/fixed-density k-grid for resolution-sensitivity benchmarks",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="per-stage timeout in seconds; 0 disables the timeout",
    )
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument(
        "--qe-version-timeout",
        type=int,
        default=60,
        help="timeout in seconds for the non-scientific pw.x version probe",
    )
    parser.add_argument(
        "--qe-nscf-mpi-processes",
        type=int,
        default=1,
        help="MPI processes/k-point pools for QE NSCF only; each rank uses one OpenMP thread",
    )
    parser.add_argument(
        "--keep-qe-tmp",
        action="store_true",
        help="retain QE wavefunction/restart scratch after writing summary.json",
    )
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("--timeout must be nonnegative; use 0 to disable it")
    if args.timeout == 0:
        args.timeout = None
    if args.ecutwfc_override is not None and args.ecutwfc_override <= 0:
        parser.error("--ecutwfc-override must be positive")
    if args.ecutrho_override is not None and args.ecutrho_override <= 0:
        parser.error("--ecutrho-override must be positive")
    if args.conv_thr_override is not None and args.conv_thr_override <= 0:
        parser.error("--conv-thr-override must be positive")
    if args.qe_nscf_mpi_processes < 1:
        parser.error("--qe-nscf-mpi-processes must be positive")
    if args.qe_version_timeout < 1:
        parser.error("--qe-version-timeout must be positive")
    if (
        args.ecutwfc_override is not None
        and args.ecutrho_override is not None
        and args.ecutrho_override < args.ecutwfc_override
    ):
        parser.error("--ecutrho-override must be >= --ecutwfc-override")
    for name, grid in (
        ("--scf-kgrid-override", args.scf_kgrid_override),
        ("--band-kgrid-override", args.band_kgrid_override),
    ):
        if grid is not None and any(value < 1 for value in grid):
            parser.error(f"{name} values must all be >= 1")
    if args.single:
        if args.row_index is None or args.row_json is None:
            parser.error("--single requires --row-index and --row-json")
        return single(args)
    parser.error("use --single; job dispatch is managed by the validation pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
