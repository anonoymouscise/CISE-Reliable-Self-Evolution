from __future__ import annotations
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from pymatgen.core import Composition, Lattice, Structure

PV_ALLOWED = set('H Li B C O F Na Mg Al Si P S Cl K Ca Sc Ti V Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb La Ce Nd'.split())
SSE_REQUIRED = set('Li Na K Mg Ca Al'.split())


GROUPS = [('Li','Na','K','Rb'), ('Mg','Ca','Sr','Ba','Zn'), ('Al','Ga','In'),
          ('Si','Ge'), ('Ti','Zr','Hf'), ('O','S','Se'), ('F','Cl','Br','I'), ('N','P','As')]
RADII = dict(zip('Li Na K Rb Mg Ca Sr Ba Zn Al Ga In Si Ge Ti Zr Hf O S Se F Cl Br I N P As Cu'.split(),
                [1.28,1.66,2.03,2.20,1.41,1.76,1.95,2.15,1.22,1.21,1.22,1.42,1.11,1.20,1.60,1.75,1.75,.66,1.05,1.20,.57,1.02,1.20,1.39,.71,1.07,1.19,1.32]))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _meta_path(path):
    return Path(str(path) + '.meta.json')

def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def _geometry(structure):
    if not structure.is_ordered or not 1 <= len(structure) <= 80:
        raise ValueError('requires_ordered_1_to_80_sites')
    lattice = np.asarray(structure.lattice.matrix)
    if not np.isfinite(lattice).all() or np.linalg.det(lattice) <= 0:
        raise ValueError('requires_finite_positive_volume_cell')
    if not np.isfinite(structure.frac_coords).all():
        raise ValueError('nonfinite_coordinates')
    volume_per_atom = structure.volume / len(structure)
    if not 3.0 <= volume_per_atom <= 80.0:
        raise ValueError('heuristic_volume_per_atom_outside_3_to_80_A3')


    translations = np.array([v for v in itertools.product(range(-2,3), repeat=3) if any(v)])
    self_distance = float(np.linalg.norm(translations @ lattice, axis=1).min())
    distances = structure.distance_matrix
    np.fill_diagonal(distances, np.inf)
    minimum = min(self_distance, float(distances.min()))
    bonds = {}
    for i, site in enumerate(structure):
        a = site.specie.symbol
        if self_distance < max(.8, .65 * (2 * RADII.get(a,1.2))):
            raise ValueError('heuristic_self_image_overlap:' + a)
        for j in range(i+1, len(structure)):
            b = structure[j].specie.symbol
            d = float(distances[i,j])
            cutoff = max(.8, .65 * (RADII.get(a,1.2) + RADII.get(b,1.2)))
            if d < cutoff:
                raise ValueError(f'heuristic_pair_overlap:{a}-{b}:{d:.4f}<{cutoff:.4f}')
            pair = tuple(sorted((a,b)))
            bonds[pair] = min(d, bonds.get(pair, float('inf')))
    return {'min_periodic_distance_A':minimum, 'volume_per_atom_A3':volume_per_atom,
            'nearest_bonds':[{'pair':list(k),'distance_A':v} for k,v in sorted(bonds.items())]}

def task_errors(task, structure):
    elements = set(e.symbol for e in structure.composition.elements)
    if task not in ('wbg','sse','pv'):
        return ['unknown_task']
    if task == 'sse' and not elements & SSE_REQUIRED:
        return ['missing_required_mobile_ion_element']
    if task == 'pv' and not elements <= PV_ALLOWED:
        return ['outside_original_pv_allowed_elements:' + ','.join(sorted(elements-PV_ALLOWED))]
    return []

def allowed_substitutions(structure, task=None):
    present = {e.symbol for e in structure.composition.elements}
    result = {}
    for source in sorted(present):
        targets = {e for group in GROUPS if source in group for e in group} - present
        if task == 'pv':
            targets &= PV_ALLOWED
        result[source] = sorted(targets)
    return result

def parent_descriptor(path, parent_metadata=None, task=None):
    path = Path(path)
    structure = Structure.from_file(path)
    meta = parent_metadata or (json.loads(_meta_path(path).read_text()) if _meta_path(path).exists() else {})
    geom = _geometry(structure)
    root_volume = float(meta.get('root_volume_A3', structure.volume))
    return {'parent_id':meta.get('candidate_id',meta.get('parent_id',path.stem)),
            'path':str(path.resolve()), 'sha256':sha(path), 'formula':structure.composition.reduced_formula,
            'n_sites':len(structure), 'prototype_id':meta.get('prototype_id',path.stem),
            'root_volume_A3':root_volume, 'cumulative_volume_scale':structure.volume/root_volume,
            'lattice_parameters':dict(zip(('a','b','c','alpha','beta','gamma'),map(float,structure.lattice.parameters)),volume=float(structure.volume)),
            'species_sites':{s:[i for i,site in enumerate(structure) if site.specie.symbol==s] for s in sorted(set(str(s.specie) for s in structure))},
            'sites':[{'index':i,'element':site.specie.symbol,'fractional_coordinates':list(map(float,site.frac_coords))} for i,site in enumerate(structure)],
            'allowed_substitutions':allowed_substitutions(structure,task), **geom,
            'construction_status':'idealized_or_edited_unrelaxed_structure; geometry checks are heuristic'}

def prepare_library(root):

    dest = Path(root)/'prototypes'
    dest.mkdir(parents=True, exist_ok=True)
    seeds = {
        'rocksalt_MgO':Structure.from_spacegroup(225,Lattice.cubic(4.212),['Mg','O'],[[0,0,0],[.5,.5,.5]]),
        'rocksalt_NaCl':Structure.from_spacegroup(225,Lattice.cubic(5.64),['Na','Cl'],[[0,0,0],[.5,.5,.5]]),
        'zincblende_ZnS':Structure.from_spacegroup(216,Lattice.cubic(5.409),['Zn','S'],[[0,0,0],[.25,.25,.25]]),
        'fluorite_CaF2':Structure.from_spacegroup(225,Lattice.cubic(5.462),['Ca','F'],[[0,0,0],[.25,.25,.25]]),
        'antifluorite_Li2O':Structure.from_spacegroup(225,Lattice.cubic(4.62),['O','Li'],[[0,0,0],[.25,.25,.25]]),
        'perovskite_SrTiO3':Structure(Lattice.cubic(3.905),['Sr','Ti','O','O','O'],[[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]]),
        'rutile_TiO2':Structure.from_spacegroup(136,Lattice.tetragonal(4.594,2.959),['Ti','O'],[[0,0,0],[.305,.305,0]]),
        'wurtzite_ZnO':Structure.from_spacegroup(186,Lattice.hexagonal(3.25,5.207),['Zn','O'],[[1/3,2/3,0],[1/3,2/3,.382]]),
        'zincblende_GaN':Structure.from_spacegroup(216,Lattice.cubic(4.50),['Ga','N'],[[0,0,0],[.25,.25,.25]]),
        'zincblende_GaP':Structure.from_spacegroup(216,Lattice.cubic(5.451),['Ga','P'],[[0,0,0],[.25,.25,.25]]),
        'cuprite_Cu2O':Structure.from_spacegroup(224,Lattice.cubic(4.269),['Cu','O'],[[.25,.25,.25],[0,0,0]]),
        'diamond_Si':Structure.from_spacegroup(227,Lattice.cubic(5.431),['Si'],[[0,0,0]]),
    }
    descriptors = []
    for name,structure in seeds.items():
        _geometry(structure)
        path = dest/(name+'.cif')
        if path.exists():
            raise FileExistsError('refusing_to_overwrite_prototype:' + str(path))
        structure.to(filename=str(path),fmt='cif',symprec=None)
        meta = {'candidate_id':name,'prototype_id':name,'root_volume_A3':float(structure.volume),
                'provenance':'handchosen task-informed analytic crystallographic prototype with fixed approximate lattice constants; not relaxed; no dataset property-label filtering',
                'library_version':1,'builder_sha256':sha(__file__)}
        _write(_meta_path(path),meta)
        descriptor = parent_descriptor(path)
        descriptor['eligible_tasks'] = [t for t in ('wbg','sse','pv') if not task_errors(t,structure)]
        descriptors.append(descriptor)
    _write(dest/'manifest.json',descriptors)
    return descriptors

def apply_edit(parent_path, proposal, task, output_path, parent_metadata=None):

    parent_path, output_path = Path(parent_path), Path(output_path)
    parent = parent_descriptor(parent_path,parent_metadata,task)
    if proposal.get('parent_id') != parent['parent_id']:
        raise ValueError('parent_id_mismatch')
    substitutions = proposal.get('substitutions')
    if not isinstance(substitutions,list) or len(substitutions)>2:
        raise ValueError('zero_to_two_whole_species_substitutions_required')
    replacements = {}
    for edit in substitutions:
        if not isinstance(edit,dict) or set(edit)!={'from_element','to_element'}:
            raise ValueError('invalid_substitution_schema')
        source,target = edit['from_element'],edit['to_element']
        if source in replacements or target not in parent['allowed_substitutions'].get(source,[]):
            raise ValueError('substitution_not_allowed_or_duplicate_source')
        replacements[source] = target
    factor = proposal.get('lattice_scale')
    if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not .95<=factor<=1.05:
        raise ValueError('lattice_scale_must_be_0.95_to_1.05')
    if not replacements and abs(factor-1)<1e-6:
        raise ValueError('identity_edit_not_allowed')
    structure = Structure.from_file(parent_path)
    species = [replacements.get(site.specie.symbol,site.specie.symbol) for site in structure]
    child = Structure(structure.lattice.matrix * factor, species, structure.frac_coords, to_unit_cell=True)
    cumulative = child.volume / parent['root_volume_A3']
    if not .75 <= cumulative <= 1.25:
        raise ValueError('cumulative_volume_outside_0.75_to_1.25_of_prototype')
    errors = task_errors(task,child)
    if errors:
        raise ValueError(';'.join(errors))
    geometry = _geometry(child)
    if output_path.exists() or _meta_path(output_path).exists():
        raise FileExistsError('refusing_to_overwrite_candidate')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    child.to(filename=str(output_path),fmt='cif',symprec=None)
    meta = {'candidate_id':proposal.get('candidate_id',output_path.stem),'prototype_id':parent['prototype_id'],
            'root_volume_A3':parent['root_volume_A3'],'parent_id':parent['parent_id'],
            'parent_path':parent['path'],'parent_sha256':parent['sha256'],
            'substitutions':substitutions,'lattice_scale':factor,'task':task,
            'same_composition_as_parent':child.composition.reduced_composition==Structure.from_file(parent_path).composition.reduced_composition,
            'builder_sha256':sha(__file__),'geometry_validation':geometry,
            'novelty_scope':'generated edit; global/database novelty unverified'}
    _write(_meta_path(output_path),meta)
    result = parent_descriptor(output_path,meta,task)
    result.update({'lineage':meta,'cif_path':str(output_path.resolve()),'cif_sha256':sha(output_path),
                   'formula':child.composition.reduced_formula})
    return result
