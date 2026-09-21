import subprocess, re, time, sys
from concurrent.futures import ThreadPoolExecutor
CAP = 60
seeds = [l.strip() for l in open('results/est149_seeds.txt') if l.strip()]
out = open('results/est149_probe.tsv', 'w'); out.write("seed\tstatus\telapsed\tstates\tbest\n"); out.flush()
def run(seed):
    t0 = time.time()
    r = subprocess.run(['./backsearch_worker_nt', '--grid', '5x5', '--two-tables', '--allow-exit-transit', '--num-holes', '2',
                        '--exit', '1', '--seed-path', seed, '--time', str(CAP)], capture_output=True, text=True).stdout
    st = re.search(r'--- Exit 1 \((.*?)\)', r); el = re.search(r'elapsed:\s+([\d.]+)', r)
    ns = re.search(r'states checked: (\d+)', r); b = re.search(r'best depth:\s+(\d+)', r)
    elapsed = el.group(1) if el else f"{time.time()-t0:.3f}"
    return f"{seed}\t{st.group(1) if st else 'rc'}\t{elapsed}\t{ns.group(1) if ns else -1}\t{b.group(1) if b else 0}\n"
with ThreadPoolExecutor(2) as ex:
    for line in ex.map(run, seeds): out.write(line); out.flush()
print("done")
