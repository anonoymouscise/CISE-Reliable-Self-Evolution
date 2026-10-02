import re, math


TASK_CONSTRAINTS = {'Stable Wide-Bandgap Semiconductors': {'numeric': [('band_gap', '>=', 2.5),
                                                    ('formation_energy',
                                                     '<=',
                                                     -1.0)]},
 'Solid-State Electrolytes': {'numeric': [('formation_energy', '<=', -1.0),
                                          ('band_gap', '>=', 2.0)],
                              'categorical': {'requires_any_element': [['Li'],
                                                                       ['Na'],
                                                                       ['K'],
                                                                       ['Mg'],
                                                                       ['Ca'],
                                                                       ['Al']]}},
 'Photovoltaic Absorbers': {'numeric': [('band_gap', 'in', (0.7, 2.0)),
                                        ('formation_energy', '<=', 0.0)],
                            'categorical': {'earth_abundant': True,
                                            'non_toxic': True}}}


TARGET_OVERRIDES = {

    "energy_above_hull": (0.0, 5.0),
}

TASK_OBJECTIVES = {'Stable Wide-Bandgap Semiconductors': {'lower': ['formation_energy'],
                                        'higher': ['band_gap']},
 'Solid-State Electrolytes': {'lower': ['formation_energy'],
                              'higher': ['band_gap']},
 'Photovoltaic Absorbers': {'lower': ['formation_energy'],
                            'higher': ['band_gap']}}


PROPERTY_SET = {'formation_energy', 'band_gap'}
ALIASES = {'formation_energy': 'formation_energy',
 'band_gap': 'band_gap',
 'formation energy': 'formation_energy',
 'e_form': 'formation_energy',
 'ef': 'formation_energy',
 'band gap': 'band_gap',
 'bandgap': 'band_gap',
 'e_g': 'band_gap'}
def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)


def _infer_prop_from_stdout(stdout: str) -> str | None:
    s = (stdout or "").lower()
    for key, prop in [("formation energy", "formation_energy"),
                      ("formation-energy", "formation_energy"), ("e_form", "formation_energy"),
                      ("band gap", "band_gap"), ("bandgap", "band_gap")]:
        if key in s:
            return prop
    return None

def _first_number(text: str):
    if not text: return None
    m = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(text))
    return float(m[0]) if m else None

def _values_from_predictions(predictions: dict) -> dict:

    sums, counts = {}, {}
    for model_name, res in (predictions or {}).items():
        v, stdout = None, None
        if isinstance(res, dict):
            for k in ("value","pred","prediction","y","score"):
                if k in res and isinstance(res[k], (int,float)):
                    v = float(res[k]); break
            stdout = res.get("stdout")
            if v is None and stdout:
                v = _first_number(stdout)
        if v is None:
            continue


        prop = _canon(model_name)
        if prop not in PROPERTY_SET:
            lowered_name = str(model_name).strip().lower()
            for model_prefix in ("cgcnn_", "alignn_", "matgl_", "m3gnet_"):
                if lowered_name.startswith(model_prefix):
                    prop = _canon(lowered_name[len(model_prefix):])
                    break
        if prop not in PROPERTY_SET:
            prop = _infer_prop_from_stdout(stdout) or prop
        if prop not in PROPERTY_SET:
            continue

        sums[prop]   = sums.get(prop, 0.0) + v
        counts[prop] = counts.get(prop, 0) + 1

    return {p: sums[p]/counts[p] for p in sums}


def _violations_for(task: str, vals: dict, candidate: dict | None):
    spec = TASK_CONSTRAINTS.get(task, {})
    vios = []


    for prop, op, th in spec.get("numeric", []):
        v = vals.get(prop)
        ok = False

        if prop == "electrical_conductivity" and v is not None:

            adjusted_v = v / 10000
            if op == ">=":
                ok = adjusted_v >= th
            elif op == "<=":
                ok = adjusted_v <= th
            elif op == "in":

                lo, hi = th
                adjusted_lo = lo / 10000
                adjusted_hi = hi / 10000
                ok = adjusted_lo <= adjusted_v <= adjusted_hi
            elif op == "≈":
                target, tol = th
                adjusted_target = target / 10000
                adjusted_tol = tol / 10000
                ok = abs(adjusted_v - adjusted_target) <= adjusted_tol
        else:

            if prop in TARGET_OVERRIDES:
                target, tol = TARGET_OVERRIDES[prop]
                ok = (v is not None) and (abs(v - target) <= tol)
            else:
                if op == ">=":
                    ok = (v is not None) and (v >= th)
                elif op == "<=":
                    ok = (v is not None) and (v <= th)
                elif op == "in":
                    lo, hi = th; ok = (v is not None) and (lo <= v <= hi)
                elif op == "≈":
                    target, tol = th; ok = (v is not None) and (abs(v - target) <= tol)
        if not ok:
            vios.append((prop, op, th, v))


    if spec.get("categorical"):
        sp = set()
        if candidate:
            if isinstance(candidate.get("species"), list):
                sp = {str(x) for x in candidate["species"]}
            elif isinstance(candidate.get("formula"), str):
                sp = set(re.findall(r"[A-Z][a-z]?", candidate["formula"]))
        cat = spec["categorical"]
        need_any = cat.get("requires_any_element", [])
        if need_any:
            ok_any = any(any(el in sp for el in group) for group in need_any)
            if not ok_any:
                vios.append(("composition", "requires_any_of", need_any, sp))
        if cat.get("non_toxic"):

            toxic_elements = {"Hg","Cd","Pb","Tl","As","Be","Se","U","Th","Sb"}
            if any(e in sp for e in toxic_elements):
                vios.append(("composition","non_toxic",None,sp))
        if cat.get("earth_abundant"):

            earth_abundant_elements = {"H","Li","B","C","O","F","Na","Mg","Al","Si","P","S","Cl","K","Ca","Sc","Ti","V","Mn","Fe","Co","Ni","Cu","Zn","Ga","Rb","Sr","Y","Zr","Nb","La","Ce","Nd"}

            if any(e not in earth_abundant_elements for e in sp):
                vios.append(("composition","earth_abundant",None,sp))

    return vios

def _threshold_def(task: str, prop: str):


    if prop in TARGET_OVERRIDES:
        return ("approx", TARGET_OVERRIDES[prop])

    for p, op, th in TASK_CONSTRAINTS.get(task, {}).get("numeric", []):
        if p == prop:
            if op == ">=": return ("ge", th)
            if op == "<=": return ("le", th)
            if op == "in": return ("in", th)
            if op == "≈":  return ("approx", th)
    return None

def _margin(kind, th, direction, value):
    if value is None:
        return 0.0
    if kind == "ge":
        T = th
        return min(value - T, 10.0) if direction == "higher" else min(T - value, 10.0)
    if kind == "le":
        T = th
        return min(T - value, 10.0) if direction == "lower" else min(value - T, 10.0)
    if kind == "in":
        lo, hi = th
        return min(value - lo, hi - value)
    if kind == "approx":
        target, tol = th
        return (tol - abs(value - target))
    return 0.0

def simple_score(predictions, _constraints_unused, task_name=None, candidate=None, print_model_scores=False):


    vals = _values_from_predictions(predictions)
    if print_model_scores:
        print(f"[Agent] Parsed values (by property): {vals}")


    vios = _violations_for(task_name or "", vals, candidate)
    total_constraints = len(TASK_CONSTRAINTS.get(task_name or "", {}).get("numeric", []))


    if TASK_CONSTRAINTS.get(task_name or "", {}).get("categorical"):
        cat_constraints = TASK_CONSTRAINTS[task_name]["categorical"]
        if cat_constraints.get("requires_any_element"):
            total_constraints += 1
        if cat_constraints.get("non_toxic"):
            total_constraints += 1
        if cat_constraints.get("earth_abundant"):
            total_constraints += 1

    failed_constraints = len(vios)
    passed_constraints = total_constraints - failed_constraints

    if print_model_scores:
        print(f"[Agent] Constraint analysis: {passed_constraints}/{total_constraints} passed, {failed_constraints} failed")
        for (p, op, th, v) in vios:
            print(f"[Agent] VIOLATION: {p} {op} {th} (got {v})")


    if failed_constraints == total_constraints:

        if print_model_scores:
            print(f"[Agent] All constraints failed → score = -10")
        return -10.0
    elif failed_constraints > 0:

        if print_model_scores:
            print(f"[Agent] Partial constraint passing → score = 0")
        return 0.0
    else:


        def _diff_from_thdef(kind, th, value):
            if value is None:
                return None
            if kind == "ge":
                T = th
                return max(0.0, T - value)
            if kind == "le":
                T = th
                return max(0.0, value - T)
            if kind == "in":
                lo, hi = th
                if lo <= value <= hi:
                    return 0.0
                return min(abs(value - lo), abs(value - hi))
            if kind == "approx":


                target, _tol_unused = th
                return abs(value - target)
            return None


        obj = TASK_OBJECTIVES.get(task_name or "", {"lower": [], "higher": []})


        props = []
        for p in vals.keys():
            if _threshold_def(task_name or "", p) is not None:
                props.append(p)
        if not props:
            if print_model_scores:
                print(f"[Agent] No usable properties for scoring → score = 1")
            return 1.0

        per_prop_scores = []
        for p in props:
            thdef = _threshold_def(task_name or "", p)
            if thdef is None:
                continue


            direction = "lower" if p in obj.get("lower", []) else ("higher" if p in obj.get("higher", []) else None)


            value = vals.get(p)
            scaled_thdef = thdef
            if p == "electrical_conductivity" and value is not None:
                value = value / 10000

                if thdef[0] == "in":
                    lo, hi = thdef[1]
                    scaled_thdef = (thdef[0], (lo / 10000, hi / 10000))
                elif thdef[0] in ["ge", "le", "≈"]:
                    th_val = thdef[1]
                    if thdef[0] == "≈":
                        target, tol = th_val
                        scaled_thdef = (thdef[0], (target / 10000, tol / 10000))
                    else:
                        scaled_thdef = (thdef[0], th_val / 10000)

            if p in TARGET_OVERRIDES:

                diff = _diff_from_thdef(scaled_thdef[0], scaled_thdef[1], value)
                if diff is None:
                    continue
                score_p = 10.0 / (1.0 + float(diff))
                if score_p > 10.0:
                    score_p = 10.0
                per_prop_scores.append(score_p)
                if print_model_scores and p in vals:
                    print(f"[Agent] {p}: value={vals[p]} vs {thdef[0]} {thdef[1]} diff={diff:.4f} -> inv={score_p:.4f} (cap 10)")
            else:


                inferred_direction = direction
                if inferred_direction is None:
                    if scaled_thdef[0] == "ge":
                        inferred_direction = "higher"
                    elif scaled_thdef[0] == "le":
                        inferred_direction = "lower"
                    else:
                        inferred_direction = "higher"
                contrib = _margin(scaled_thdef[0], scaled_thdef[1], inferred_direction, value)
                per_prop_scores.append(contrib)
                if print_model_scores and p in vals:
                    print(f"[Agent] {p}: value={vals[p]} vs {thdef[0]} {thdef[1]} -> margin={contrib:+.4f}")


        if not per_prop_scores:
            if print_model_scores:
                print(f"[Agent] No usable properties for scoring → score = 1")
            return 1.0

        final_score = sum(per_prop_scores) / len(per_prop_scores)
        if final_score > 10.0:
            final_score = 10.0
        if print_model_scores:
            print(f"[Agent] Equal-weight mean score over {len(per_prop_scores)} props (capped at 10): {final_score:.4f}")
        return final_score
