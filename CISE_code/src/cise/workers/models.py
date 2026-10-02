import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import traceback
import zipfile
from cise.runtime import CONFIG

def main(kind):
    config=CONFIG['surrogates'];device=config['device']
    with contextlib.redirect_stdout(sys.stderr):
        if kind=='mgt':
            os.environ['CISE_MGT_BUNDLE']=config['mgt_bundle']


            from .mgt_adapter import load_model,predict_band_gap
            import torch
            torch.set_num_threads(1)
            model=load_model(device)
        else:
            import torch
            torch.set_num_threads(1)
            root=Path(config['alignn_root']);sys.path.insert(0,str(root))
            from alignn.models.alignn import ALIGNN,ALIGNNConfig
            from alignn.graphs import Graph
            from jarvis.core.atoms import Atoms
            models={}
            for prop,name in [('formation_energy','jv_formation_energy_peratom_alignn')]:
                with zipfile.ZipFile(root/'alignn'/(name+'.zip')) as z:
                    cfg=next(n for n in z.namelist() if 'config.json' in n)
                    cp=[n for n in z.namelist() if ('checkpoint_' in n and 'pt' in n) or 'best_model.pt' in n][-1]
                    m=ALIGNN(ALIGNNConfig(**json.loads(z.read(cfg))['model']))

                    m.load_state_dict(torch.load(io.BytesIO(z.read(cp)),map_location='cpu',weights_only=False)['model'],strict=True)
                    models[prop]=m.eval()
    print(json.dumps({'ready':kind}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
                if kind=='mgt':values={'band_gap':predict_band_gap(request['cif'],device,model)}
                else:
                    atoms=Atoms.from_cif(request['cif']);g,lg=Graph.atom_dgl_multigraph(atoms,cutoff=8.,max_neighbors=12)
                    values={p:float(models[p]([g,lg,torch.tensor(atoms.lattice_mat)]).reshape(-1)[0]) for p in request.get('properties',['formation_energy'])}
                if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite prediction')
            result=dict(status='completed',values=values)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result=dict(status='failed',error_type=type(exc).__name__)
        print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['mgt','alignn']);main(ap.parse_args().kind)
