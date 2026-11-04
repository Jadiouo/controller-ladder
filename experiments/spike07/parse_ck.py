import sys, re, ast, glob
for f in sorted(glob.glob(sys.argv[1])):
    cks = [ast.literal_eval(l.strip()[len('ckpt '):]) for l in open(f) if l.strip().startswith('ckpt ')]
    tr = [l for l in open(f) if 'train-frac_cap' in l]
    print(f.split('/')[-1], ' '.join(f"{c['frac_cap']*100:.0f}%/{c['res_mean']:.2f}Nm" for c in cks), '| last train:', re.search(r'train-frac_cap\(last60\) (\S+)', tr[-1]).group(1) if tr else '')
