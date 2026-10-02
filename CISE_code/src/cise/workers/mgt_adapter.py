from pathlib import Path
import functools
import importlib.util
import json
import sys
import warnings
import numpy as np
import torch
import yaml
from pymatgen.core import Structure
from jarvis.core.atoms import Atoms

import os
OUT = Path(os.environ["CISE_MGT_BUNDLE"])
sys.path.insert(0, str(OUT/'repos/MGT'))
warnings.filterwarnings('ignore', message='The TorchScript type system')
spec = importlib.util.spec_from_file_location('mgt_official_graph', OUT/'upstream_preprocessing/cif2dataset_finetune_megnet.py')
GRAPH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GRAPH)

GRAPH._get_attribute_lookup = functools.lru_cache()(GRAPH._get_attribute_lookup)


def graph_from_atoms(atoms):
    return GRAPH.atom_multigraph(atoms, cutoff=4.0, max_neighbors=25,
                                 atom_features='atomic_number', compute_line_graph=False,
                                 use_canonize=False, use_lattice=False, use_angle=False)


def graph_from_cif(path):
    structure = Structure.from_file(path)
    atoms = Atoms(lattice_mat=structure.lattice.matrix, coords=structure.frac_coords,
                  elements=[str(s) for s in structure.species], cartesian=False)
    assert np.allclose(atoms.lattice_mat, structure.lattice.matrix, atol=1e-10)
    assert np.allclose(atoms.frac_coords, structure.frac_coords, atol=1e-10)
    assert atoms.num_atoms == len(structure)
    graph = graph_from_atoms(atoms)
    assert graph.x.shape == (len(structure), 92)
    for name in ['x', 'edge_attr', 'edge_nei_angle', 'edge_nei_len']:
        assert torch.isfinite(getattr(graph, name)).all(), name
    return graph


def load_model(device='cuda:0'):
    from models.mgt import MGTransformer
    config = yaml.safe_load((OUT/'repos/MGT/config/finetune.yml').read_text())
    config.update(device=device, target='gap pbe', task='finetune')
    model = MGTransformer(config, config['model'])
    state = torch.load(OUT/'repos/MGT/ckpt/finetuned/gap pbe/gap pbe_checkpoint_best.pt',
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict_band_gap(cif_path, device='cuda:0', model=None):

    from torch_geometric.data import Batch
    validation = json.loads((OUT/'mgt_validation.json').read_text())
    assert validation['valid_for_comparison']
    scaler = json.loads((OUT/'mgt_normalizer.json').read_text())
    model = load_model(device) if model is None else model
    graph = Batch.from_data_list([graph_from_cif(cif_path)]).to(device)
    with torch.no_grad():
        z = float(model(graph, graph).detach().cpu().flatten()[0])
    return z*scaler['std']+scaler['mean']


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--cif', required=True)
    args.add_argument('--device', default='cuda:0')
    options = args.parse_args()
    torch.set_num_threads(1)
    print(json.dumps({'model': 'MGT', 'band_gap_eV': predict_band_gap(options.cif, options.device)}))
