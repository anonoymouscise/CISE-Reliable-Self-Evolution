"""Fixed-occupation band-gap validation."""

import os
import re

from . import qe_runner as base

original_run=base.run_qe

def fixed_gap(text,electrons):
    occupied=int(round(electrons/2))
    blocks=[]; current=None
    for line in text.splitlines():
        if 'bands (ev)' in line.lower():
            current=[]; blocks.append(current); continue
        if current is None: continue
        stripped=line.strip()
        if stripped and re.fullmatch(r'[\s\d.+EeDd-]+',stripped):


            numbers=re.findall(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?',stripped)
            current.extend(float(x.replace('D','E').replace('d','e')) for x in numbers)
        elif current:
            current=None
    good=[x for x in blocks if len(x)>occupied]
    if not good or len(good)!=len(blocks): return {'band_gap_eV':None,'method':'fixed_eigenvalue_parse_failed'}
    vbm=max(x[occupied-1] for x in good); cbm=min(x[occupied] for x in good)
    return {'band_gap_eV':max(0.,cbm-vbm),'vbm_eV':vbm,'cbm_eV':cbm,
            'method':'fixed_occupation_electron_count_edges','valid_kpoint_blocks':len(good)}

def fixed_input(text):
    lines=[]
    for line in text.splitlines():
        if re.match(r'\s*(smearing|degauss)\s*=',line): continue
        if re.match(r'\s*occupations\s*=',line): line="  occupations = 'fixed',"
        lines.append(line)
    return '\n'.join(lines)+'\n'

def run(args,row,run_dir):
    record=original_run(args,row,run_dir)
    if args.skip_bands or not record.get('scf_converged'): return record
    record['smearing_band_gap']=record.get('band_gap')
    params=record['parameters']
    electrons=float(params['valence_electrons'])
    eligible=not params['magnetism']['spin_polarized'] and abs(electrons/2-round(electrons/2))<1e-8
    record['gap_validation_status']='UNRESOLVED_SPIN_OR_ODD_ELECTRONS' if not eligible else 'PENDING_FIXED_OCCUPATION_NSCF'
    env={**os.environ,'OMP_NUM_THREADS':str(args.omp_threads),'CUDA_VISIBLE_DEVICES':''}
    if eligible:
        nscf=run_dir/'nscf.in'
        if not nscf.exists():
            record['gap_validation_status']='MISSING_NSCF_INPUT'
            return record
        (run_dir/'nscf_fixed.in').write_text(fixed_input(nscf.read_text()))
        outcome=base.run_to_file([record['pw_x'],'-in','nscf_fixed.in'],run_dir,run_dir/'nscf_fixed.out',args.timeout,env)
        text=(run_dir/'nscf_fixed.out').read_text(errors='replace')
        gap=fixed_gap(text,electrons)
        record['fixed_nscf_run']=outcome
        record['fixed_occupation_band_gap']=gap
        if outcome['returncode']==0 and 'JOB DONE.' in text and gap.get('band_gap_eV') is not None:
            record['gap_validation_status']='FIXED_OCCUPATION_NSCF_COMPLETED'
            record['band_gap']=gap

            if record.get('energy_eV_atom') is not None: record['status']='completed'
        else:
            record['gap_validation_status']='FIXED_OCCUPATION_NSCF_FAILED_OR_UNRESOLVED'
    return record
