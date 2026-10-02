import os, math, re, fcntl, json
from pathlib import Path
import numpy as np
from .native import qe_runner as base
from .native import fixed_occupation as wrapper

if True:
    original_parameters=base.protocol_parameters
    original_input=base.qe_input
    def parameters(structure,row,pseudo_map,*args,**kwargs):
        p=original_parameters(structure,row,pseudo_map,*args,**kwargs)

        inv=np.linalg.inv(np.asarray(structure.lattice.matrix)/0.529177210903).T
        mesh=[]
        for v in inv:
            k=int(round(55*float(np.linalg.norm(v))))
            mesh.append(min(14,max(2,k+k%2)))
        p.update(xc='PBEsol',ecutwfc_Ry=45.,ecutrho_Ry=250.,cutoff_policy='ThreeBodyTB_pinned_template_45_250',
                 scf_kgrid=mesh,band_kgrid=mesh,scf_length_A=55*.529177210903,band_length_A=55*.529177210903,
                 kgrid_policy='ThreeBodyTB_even_round_55_per_bohr_cap14',occupations='Gaussian 0.01 Ry',
                 magnetism={'spin_polarized':False,'source_moment_muB_cell':None,'source_moment_field':None,
                            'initial_moment_muB_cell':0.,'initial_atomic_moments_muB':[0.]*len(structure),
                            'qe_starting_magnetization':{},'initialization':'QETB source nonspin protocol'},
                 nbands=math.ceil(p['valence_electrons']/2)+max(8,math.ceil(.2*p['valence_electrons']/2)),
                 conv_thr=1e-9,source_reconstruction=True)
        return p
    def qe_input(*args,**kwargs):
        text=original_input(*args,**kwargs)
        text=text.replace("  smearing = 'fd',","  smearing = 'gaussian',")
        text=re.sub(r'  degauss = [^\n]+',"  degauss = 0.01,",text)
        text=text.replace('  ibrav = 0,',"  ibrav = 0,\n  input_dft = 'PBEsol',\n  q2sigma = 1,\n  ecfixed = 44.5,\n  qcutz = 800,\n  force_symmorphic = .true.,")
        text=text.replace('&ELECTRONS',"&ELECTRONS\n  conv_thr = 1d-9,")
        if 'mixing_mode' not in text:text=text.replace('&ELECTRONS',"&ELECTRONS\n  mixing_mode = 'local-TF',")
        if 'mixing_beta' not in text:text=text.replace('&ELECTRONS',"&ELECTRONS\n  mixing_beta = 0.3,")
        return text
    base.protocol_parameters=parameters
    base.qe_input=qe_input
base.run_qe=wrapper.run

execute_original=base.run_to_file
run_original=base.run_qe

def execute(command,run_dir,output,timeout,env):
    if '-in' in command:
        path=Path(run_dir)/command[command.index('-in')+1]
        if path.name.startswith('nscf'):
            text=path.read_text()
            text=re.sub(r'(?mi)^\s*(diagonalization|diago_thr_init|diago_cg_maxiter|diago_full_acc)\s*=.*\n','',text)
            text=text.replace('&ELECTRONS',"&ELECTRONS\n  diagonalization = 'cg',\n  diago_thr_init = 1.0d-10,\n  diago_cg_maxiter = 200,\n  diago_full_acc = .true.,")
            path.write_text(text)
    return execute_original(command,run_dir,output,timeout,env)

def run(args,row,run_dir):
    result=run_original(args,row,run_dir)
    result['interpretation']='Fixed-structure nonspin QETB PBEsol reconstruction.'
    result['numerical_amendment']='Uniform NSCF CG, diagonalization threshold1e-10 Ry, max200, full accuracy. SCF, structures, pseudopotentials, cutoffs, kgrids and references unchanged.'
    return result


single_original=base.single
def single(args):

    row=json.loads(args.row_json);index=int(args.row_index)
    label=base.safe_label(row.get('label') or row.get('jid') or f'row_{index:04d}')
    folder=Path(args.output_dir).resolve()/'profiles'/f'{index:04d}_{label}'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if (folder/'summary.json').exists():
            record=json.loads((folder/'summary.json').read_text())
            return 0 if record.get('status')=='completed' else 1
        return single_original(args)


if __name__=='__main__':
    base.run_to_file=execute
    base.run_qe=run
    base.single=single
    raise SystemExit(base.main())
