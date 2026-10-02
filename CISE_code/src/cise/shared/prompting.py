import json

TASKS = {
    "wbg": {
        "name": "Stable Wide-Bandgap Semiconductors",
        "target": "band gap >= 2.5 eV AND formation energy <= -1.0 eV/atom",
    },
    "sse": {
        "name": "Solid-State Electrolytes",
        "target": "band gap >= 2.0 eV AND formation energy <= -1.0 eV/atom; actual species must include at least one of Li, Na, K, Mg, Ca, Al",
    },
    "pv": {
        "name": "Photovoltaic Absorbers",
        "target": "0.7 <= band gap <= 2.0 eV AND formation energy <= 0.0 eV/atom; use ONLY H, Li, B, C, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, Mn, Fe, Co, Ni, Cu, Zn, Ga, Rb, Sr, Y, Zr, Nb, La, Ce, Nd",
    },
}

_EFFECT = {"type": "string", "enum": ["increase", "decrease", "uncertain", "not_targeted"]}
_SUBSTITUTION = {
    "type": "object", "additionalProperties": False,
    "required": ["from_element", "to_element"],
    "properties": {
        "from_element": {"type": "string"}, "to_element": {"type": "string"},
    },
}
CANDIDATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["parent_id", "substitutions", "lattice_scale", "rules_used", "justification", "expected_effects", "failure_modes"],
    "properties": {
        "parent_id": {"type": "string"},
        "substitutions": {"type": "array", "minItems": 0, "maxItems": 2, "items": _SUBSTITUTION},
        "lattice_scale": {"type": "number", "minimum": 0.95, "maximum": 1.05},
        "rules_used": {"type": "string"},
        "justification": {"type": "string"},
        "expected_effects": {
            "type": "object", "additionalProperties": False,
            "required": ["band_gap", "formation_energy"],
            "properties": {k: dict(_EFFECT) for k in ("band_gap", "formation_energy")},
        },
        "failure_modes": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "string"}},
    },
}
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["candidates"],
    "properties": {"candidates": {"type": "array", "minItems": 2, "maxItems": 2, "items": CANDIDATE_SCHEMA}},
}

PROPOSAL_SCHEMA = RESPONSE_SCHEMA


def build_prompt(task, parent_descriptors, feedback, iteration):

    spec = TASKS[task]
    if not parent_descriptors:
        raise ValueError("No executable parents supplied")

    parents = [{k: v for k, v in p.items() if k not in {"path", "cif_path"}} for p in parent_descriptors]
    return f"""### ROLE ###
You are an expert materials discovery assistant using crystal chemistry and prior outcomes to propose new crystalline materials.

### TASK AND GOAL ###
Task: {spec['name']}
Evolution iteration: {iteration}
Think of two new materials that may satisfy the target properties. Select a supplied parent or prototype and explain executable structure edits. Code will apply your edits and write each CIF.
Target constraints: {spec['target']}.
These original task thresholds are unchanged. For PV, the element list is the experiment's fixed eligibility rule. For SSE, ionic conductivity is not evaluated by these two proxy properties.

### SUPPLIED PARENTS / PROTOTYPES ###
{json.dumps(parents, ensure_ascii=False, separators=(',', ':'))}

### PRIOR OUTCOMES AND VALIDATOR FEEDBACK ###
{json.dumps(feedback, ensure_ascii=False, separators=(',', ':'))}
All model values and success/failure labels in this feedback are uncorrected surrogate predictions on the exact structure: MGT supplies band gap, ALIGNN supplies formation energy. They are not DFT measurements or verified successes. An unknown value is unknown. Use both passing and failing examples and any explicit validation errors to choose your next edit.

### ALLOWED EXECUTABLE DESIGN RULES ###
1. Parent/prototype-preserving whole-species substitution: replace every occurrence of a source species with a destination listed under that parent's allowed_substitutions. Use at most two substitutions, with different source species. Both mappings refer to the original parent and execute simultaneously. Curated substitutions are chemically motivated possibilities, not proof of charge balance, stable coordination, or synthesizability.
2. Bounded isotropic lattice scaling: choose lattice_scale as a linear factor from 0.95 to 1.05; fractional sites and cell angles stay fixed. The builder also limits cumulative change relative to the root prototype. A small strain can alter properties but its favorable direction is not guaranteed.
3. Combine zero to two substitutions with one lattice_scale value (1.0 means unchanged lattice). The result must differ from the parent and from the other proposal. Respect task element constraints after every species replacement. Code preserves atom count and full occupancies and rejects invalid geometry or unsupported edits.
Choose only the supplied parent_id and permitted edits. Do not invent a parent, species site, arbitrary cell, coordinates, vacancies, interstitials, partial occupancy, surface, redox stoichiometry change, or a new prototype. Do not return a CIF, formula, or numerical property predictions; code derives the resulting formula and computes the proxy properties.

### REQUIRED STRUCTURAL REASONING ###
For each proposal, give a concise 60–120 word justification anchored to the supplied parent's actual composition, lattice/bond geometry, and available proxy feedback. Identify what the edit preserves, which bond/orbital/coordination feature it changes, and why that could move each targeted property toward the allowed region. If evidence is insufficient, state uncertainty instead of claiming improvement. Explain both sides of a gap/stability trade-off when relevant. A same-group substitution alone does not establish a property trend.
Give two or three concrete, distinct failure modes: include a plausible chemistry/geometry failure and a plausible target-property or surrogate-extrapolation failure. Do not claim DFT validation, thermodynamic stability, ionic transport, or novelty from a rationale. Negative predicted formation energy is not a convex-hull stability certificate. A structure can pass the geometric filter and still fail QE.
expected_effects must describe hypothesized change relative to the chosen parent, using increase, decrease, uncertain, or not_targeted. Use not_targeted only for properties outside this task. No rewards are assigned to confident wording.

### OUTPUT FORMAT ###
Return exactly one JSON object with key candidates, containing exactly two candidates, and no markdown or surrounding text. Each candidate has exactly:
parent_id (supplied identifier),
substitutions (list of zero to two {{"from_element":"X","to_element":"Y"}} mappings),
lattice_scale (number from 0.95 to 1.05, or 1.0 to keep the cell),
rules_used (short rule names),
justification (60–120 words),
expected_effects (object with band_gap, formation_energy),
failure_modes (two or three concise strings).
"""


def validate_response(value, schema=RESPONSE_SCHEMA):

    import jsonschema
    jsonschema.Draft202012Validator(schema).validate(value)
    for candidate in value["candidates"]:
        substitutions = candidate.get("substitutions", [])
        if len({e["from_element"] for e in substitutions}) != len(substitutions):
            raise ValueError("Substitution source species must be distinct")
        if not candidate["justification"].strip() or any(not x.strip() for x in candidate["failure_modes"]):
            raise ValueError("Empty reasoning or failure mode")
    return value["candidates"]
