"""CFX steady-state acceptance and bounded RES restart policy (no subprocesses)."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import uuid

POLICY_VERSION = 1
REQUIRED = {'U-Mom', 'V-Mom', 'W-Mom', 'P-Mass', 'H-Energy', 'K-TurbKE', 'O-TurbFreq'}
ITERATION = re.compile(r'^\s*OUTER\s+LOOP\s+ITERATION\s*=\s*(\d+)', re.M | re.I)
NUMBER = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EDed][-+]?\d+)?$')
WALL = re.compile(r'A wall has been placed at portion\(s\) of an\s+(INLET|OUTLET)', re.I)
RATIO = re.compile(r'([\d.]+)%\s+of the faces,\s*([\d.]+)%\s+of the area', re.I)

def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_tail(path):
    with Path(path).open('rb') as stream:
        stream.seek(max(0, Path(path).stat().st_size - 1024 * 1024))
        return stream.read().decode('utf-8', errors='replace')

def parse_iterations(text):
    starts = list(ITERATION.finditer(text)); blocks = []
    for i, match in enumerate(starts):
        chunk = text[match.end():starts[i+1].start() if i+1 < len(starts) else len(text)]
        rms, walls, pending, in_table = {}, {}, None, False
        for line in chunk.splitlines():
            if 'Equation' in line and 'RMS Res' in line:
                in_table = True
            if 'CFD Solver finished' in line or '======' in line:
                in_table = False
            cells = [c.strip() for c in line.split('|')]
            if in_table and len(cells) >= 7 and NUMBER.fullmatch(cells[2]):
                token = cells[3].replace('D', 'E').replace('d', 'e')
                try:
                    value = float(token)
                except ValueError:
                    value = float('nan')
                rms[cells[1]] = value if math.isfinite(value) else None
            if 'WARNING #' in line:
                pending = None
            boundary = WALL.search(line)
            if boundary:
                pending = boundary.group(1).lower()
            ratio = RATIO.search(line)
            if pending and ratio:
                walls[pending] = [float(ratio[1]), float(ratio[2])]
                pending = None
        blocks.append({'iteration':int(match[1]), 'rms':rms, 'walls':walls})
    return blocks

def inspect_out(path, threshold=1e-4, wall_patience=3):
    text = read_tail(path)
    blocks = parse_iterations(text)
    last = blocks[-1] if blocks else {'iteration':None, 'rms':{}, 'walls':{}}
    complete = REQUIRED.issubset(last['rms']) and all(v is not None for v in last['rms'].values())
    maximum = max(last['rms'].values()) if complete else None
    tail = blocks[-wall_patience:]
    consecutive = len(tail) == wall_patience and all(b['iteration'] == tail[0]['iteration']+i for i,b in enumerate(tail))
    blocked = consecutive and any(all(b['walls'].get(side) == [100.,100.] for b in tail) for side in ('inlet','outlet'))
    finished = 'This run of the ANSYS CFX Solver has finished.' in text
    fatal = bool(re.search(r'FATAL OVERFLOW|floating point overflow|An error has occurred in cfx5solve',text,re.I))
    return {**last, 'out':str(path), 'max_rms':maximum, 'complete':complete, 'finished':finished,
            'blocked':bool(blocked), 'fatal':fatal, 'threshold':threshold,
            'accepted':bool(complete and finished and not blocked and not fatal and maximum < threshold)}

def latest_pair(directory):
    """The latest OUT must have its own RES. Never pair unrelated timestamps."""
    outs = sorted(Path(directory).glob('*.out'), key=lambda p:(p.stat().st_mtime_ns,p.name))
    if not outs:
        return None, None
    out = outs[-1]
    res = out.with_suffix('.res')
    return out, res if res.exists() else None

def legacy_restart_budget(directory):
    """Migrate the earlier DOE_resume script's explicit CCL iteration allocations."""
    attempts = []
    for out in sorted(Path(directory).glob('DOE_resume_*.out')):
        ccl = out.with_suffix('.ccl')
        if not ccl.exists():
            raise ValueError('Legacy restart OUT lacks its CCL budget: '+out.name)
        match = re.search(r'Maximum Number of Iterations\s*=\s*(\d+)',ccl.read_text(encoding='utf-8'),re.I)
        if not match:
            raise ValueError('Cannot recover legacy restart budget: '+out.name)
        attempts.append({'out':out.name,'iterations':int(match[1]),'migrated_from':'DOE_resume'})
    return {'policy_version':1,'extra_iterations_reserved':sum(a['iterations'] for a in attempts),'attempts':attempts}

def restart_ccl(out, iterations, threshold):
    with Path(out).open('r',encoding='utf-8',errors='replace') as stream:
        header = stream.read(2 * 1024 * 1024)
    flows = sorted(set(re.findall(r'^\s*FLOW:\s*(.+?)\s*$',header,re.M)))
    if len(flows) != 1:
        raise ValueError('Cannot uniquely identify FLOW in CFX OUT; refusing guessed restart CCL')
    return (f'FLOW: {flows[0]}\n  SOLVER CONTROL:\n    CONVERGENCE CONTROL:\n'
            f'      Maximum Number of Iterations = {int(iterations)}\n      Minimum Number of Iterations = 1\n'
            '    END\n    CONVERGENCE CRITERIA:\n      Residual Type = RMS\n'
            f'      Residual Target = {threshold * 0.1:.8g}\n    END\n  END\nEND\n')

def reject_result(directory, reason, evidence):
    """Preserve evidence but make previously extracted values unusable."""
    root = Path(directory)
    archive = root/'rejected_results'/uuid.uuid4().hex
    for name in ('CFX_Results.txt','CFX_Results.meta.json'):
        path = root/name
        if path.exists():
            archive.mkdir(parents=True,exist_ok=True)
            path.replace(archive/name)
    atomic_json(root/'CFX_INVALID.json', {'policy_version':POLICY_VERSION,'reason':reason,'evidence':evidence})
