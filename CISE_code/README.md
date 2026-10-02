# CISE: Conformal Interval-driven Self-Evolving

CISE is a framework for reliable self-evolving LLM agents that operate with imperfect proxy reward models. In many open-ended search problems, true rewards are expensive or impractical to obtain during evolution, so agents must rely on cheaper but imperfect proxy evaluations. CISE uses candidate-specific conformal intervals to account for uncertainty in these proxy rewards, guiding evolutionary feedback and selecting solutions that reliably satisfy target constraints.

This repository includes two self-evolution methods:

LLEMA: evolutionary search using point estimates from fixed proxy reward models.

CISE: evolutionary search using Gibbs conformal intervals and online density-ratio estimation. Configuration files use the method identifier cci.

## Install

Python 3.11+ on Linux/POSIX:

```bash
python -m pip install .
```

## Offline example

Synthetic predictions; no API, external models or DFT:

```bash
cise configure --method cci --demo --output demo.json
cise init --config demo.json
cise run --config demo.json
```

Use `--method llema` for the baseline.

## Search

Edit `configs/cci.json` or `configs/llema.json`. Paths are relative to the configuration file. Supply:

- MGT band-gap model bundle, including its model code, checkpoint, preprocessing and validated normalizer.
- ALIGNN source and `alignn/jv_formation_energy_peratom_alignn.zip`.
- For CISE, frozen calibration assets for `band_gap` and `formation_energy`: `<property>_scale.joblib`, `<property>_residuals.json` and task configuration files. QR mode additionally needs matching frozen quantile models. Only load trusted serialized models.

Model workers use the interpreters configured under `surrogates`; install each model's dependencies in its environment. Set `OPENROUTER_API_KEY` and configure `llm.model`.

```bash
cise doctor --config configs/cci.json
cise init --config configs/cci.json
cise run --config configs/cci.json
cise report --config configs/cci.json
```

Results are written to `output_dir`. Use one controller per output directory. Rerun the same configuration to resume; use a new output directory after changing code, configuration or assets. API spending is not capped by the package.

Optional `qe.enabled` validation requires Quantum ESPRESSO `pw.x`, GBRV PBEsol pseudopotentials, matched elemental references and source-replay evidence. Missing DFT results remain unresolved. Software demonstrations do not establish scientific coverage.

Calibration import/preparation and other commands are listed by `cise --help`. Data, model weights and experiment outputs are not included.

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
