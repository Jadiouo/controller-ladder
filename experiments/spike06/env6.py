"""spike06 environment = spike05 env5 with a configurable (wider) error distribution and an extended zeta map (e down to ~0.31).
Usage: import env6 as W; W.use(dict(e=[lo,hi], mb=[lo,hi], delay=[lo_ms,hi_ms])); then W.V is the env5 module (JugEnv5, evaluate, ...).
Plant parameters per episode still derive from the seed only (env5.draw_params), so every method sees identical errors."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'spike05'))
import env5 as V
V._ZM = json.load(open(os.path.join(HERE, 'zeta_map6.json')))
def use(d):
    V.E_RANGE = tuple(d['e']); V.MB_RANGE = tuple(d['mb']); V.DELAY_MS_RANGE = tuple(d['delay'])
def use_file(path='results/spike06/dist.json'):
    d = json.load(open(path)); use(d); return d
