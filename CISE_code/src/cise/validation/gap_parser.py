import hashlib
import math
import re
from pathlib import Path

NUM = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?'


def numeric_lines(text):
    values = []
    for line in text.splitlines():
        if not line.strip() and not values:
            continue
        if not line.strip() or not re.fullmatch(r'[\s\d.+EeDd-]+', line):
            break
        values.extend(float(x.replace('D', 'E').replace('d', 'e')) for x in re.findall(NUM, line))
    return values


def parse_gap(path, electrons, nspin, nbnd, moment=0):
    path = Path(path)
    text = path.read_text()
    assert 'JOB DONE.' in text and 'Error in routine' not in text
    assert not re.search(r'(?:bands?|eigenvalues?)\s+(?:are\s+)?not converged', text, re.I), 'eigenvalue convergence warning'
    sections = re.split(r'-+\s*SPIN (UP|DOWN)\s*-+', text)
    if nspin == 2:
        assert sections[1::2] == ['UP', 'DOWN']
        sections = dict(zip(sections[1::2], sections[2::2]))
    else:
        assert nspin == 1 and len(sections) == 1
        sections = {'PAIRED': text}
    counts = {'PAIRED': electrons/2} if nspin == 1 else {'UP': (electrons+moment)/2, 'DOWN': (electrons-moment)/2}
    assert all(abs(n-round(n)) < 1e-8 and 0 < n < nbnd for n in counts.values()), 'nonintegral spin occupation'
    nk = int(re.search(r'number of k points\s*=\s*(\d+)', text)[1])
    vbm, cbm, deviations, eigs = [], [], [], {}
    for spin, section in sections.items():
        blocks = re.split(r'bands \(ev\):[^\n]*\n', section, flags=re.I)[1:]
        assert len(blocks) == nk
        n = round(counts[spin]); eigs[spin] = []
        for block in blocks:
            eigenvalues = numeric_lines(block)
            assert len(eigenvalues) == nbnd and all(math.isfinite(x) for x in eigenvalues)
            assert all(b >= a-.00011 for a,b in zip(eigenvalues,eigenvalues[1:]))
            vbm.append(eigenvalues[n-1]); cbm.append(eigenvalues[n]); eigs[spin].append(eigenvalues)
            if 'occupation numbers' in block:
                occ = numeric_lines(block.split('occupation numbers',1)[1]); assert len(occ) == nbnd
                deviations.extend(abs(x-(1 if i<n else 0)) for i,x in enumerate(occ))
    gap = max(0., min(cbm)-max(vbm))
    spin_diff = None
    if nspin == 2:
        assert abs(moment) < 1e-8, 'magnetic solution requires separate preregistered parser'
        spin_diff = max(abs(x-y) for a,b in zip(eigs['UP'],eigs['DOWN']) for x,y in zip(a,b))
        assert spin_diff <= .0011
        if gap > 0:
            assert deviations and max(deviations) <= .00011
    return dict(band_gap_eV=gap, vbm_eV=max(vbm), cbm_eV=min(cbm),
                method='strict_all_kpoint_spin_electron_count', spatial_kpoints=nk,
                spin_difference_eV=spin_diff, source=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
