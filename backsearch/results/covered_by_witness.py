#!/usr/bin/env python3
"""covered_by_witness.py B H E B2 H2  (x = unconstrained)
Exit 0 iff the live proof for (5x5, exit E, <=B2 blocks, <=H2 holes) has a stored
champion with <= B blocks and <= H holes -- then the subclass (B, H, E) has the
same maximum and needs no run.  Exit 1 otherwise (including API failure)."""
import json, sys, urllib.request
B, H, E, B2, H2 = sys.argv[1:6]
cap = lambda v: None if v == 'x' else int(v)
try:
    proofs = json.load(urllib.request.urlopen('https://pathology.georgespahn.com/api/proofs', timeout=30))['proofs']
except Exception as ex:
    print(f"api error: {ex}"); sys.exit(1)
for p in proofs:
    if p.get('derived') or p['rows'] != 5 or p['cols'] != 5 or p['exitCell'] != int(E): continue
    if p['maxBlocks'] != cap(B2) or p['maxHoles'] != cap(H2): continue
    for c in p.get('champions') or []:
        if (cap(B) is None or c['blocks'] <= cap(B)) and (cap(H) is None or c['holes'] <= cap(H)):
            print(f"witness #{c['id']} ({c['blocks']} blocks, {c['holes']} holes, {c['moves']} moves) of proof #{p['id']}"); sys.exit(0)
    print(f"proof #{p['id']} max {p['maxMoves']} has no champion within <={B} blocks <={H} holes"); sys.exit(1)
print("no such proof"); sys.exit(1)
