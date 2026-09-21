import json, re, subprocess, time
s=json.load(open('results/camp_h1_summary.json')); log=open('results/submit_h1_champions.log','a')
for e,m in sorted(s['best'].items(), key=lambda kv:int(kv[0])):
    path=s['bestjob'][e]
    out=subprocess.run(['./backsearch_worker_nt','--grid','5x5','--two-tables','--allow-exit-transit','--num-holes','1','--exit',e,'--seed-path',path,'--time','7200'],capture_output=True,text=True).stdout
    mm=re.search(r'^%s \([^)]*\)\n((?:  .*\n)+)'%m, out, re.M)
    if not mm: log.write(f"exit {e}: could not extract {m}-move level\n"); log.flush(); continue
    code=subprocess.run(['./level_to_pathology'],input=mm.group(1),capture_output=True,text=True).stdout.strip()
    while True:
        r=subprocess.run(['curl','-s','-m','300','-D','-','-H','Content-Type: application/json','--data',json.dumps({"code":code,"attribution":"Panacea"}),'https://pathology.georgespahn.com/api/submit'],capture_output=True,text=True).stdout
        hdr,body=re.split(r'\r?\n\r?\n', r, 1) if re.search(r'\r?\n\r?\n', r) else (r,'')
        if ' 429 ' in hdr.split('\n')[0]: time.sleep(130); continue
        try: j=json.loads(body); msg=f"accepted id {j['level']['id']}" if j.get('accepted') else f"refused: {j.get('reason')}"
        except Exception: msg=body[:100]
        log.write(f"{time.strftime('%F %T')} exit {e} ({m}) {code.replace(chr(10),'/')}: {msg}\n"); log.flush(); break
    time.sleep(61)
log.write("done\n")
