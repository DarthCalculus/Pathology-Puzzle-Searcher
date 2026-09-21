import json, re, subprocess, sys, time
# Re-submit champions that hit the site's 60/hour submit limit; honour Retry-After.
rows=[re.match(r'champion (\S+) exit (\d+) \((\d+)\): (\S+) ->', l).groups() for l in open('results/classes/resubmit.txt') if l.startswith('champion')]
log=open('results/classes/resubmit.log','a')
for label,e,m,code in rows:
    code=code.replace('/','\n')
    while True:
        r=subprocess.run(['curl','-s','-m','300','-D','-','-H','Content-Type: application/json','--data',json.dumps({"code":code,"attribution":"Panacea"}),'https://pathology.georgespahn.com/api/submit'],capture_output=True,text=True).stdout
        hdr,body=re.split(r'\r?\n\r?\n', r, 1) if re.search(r'\r?\n\r?\n', r) else (r, '')
        if ' 429 ' in hdr.split('\n')[0]:
            ra=[l for l in hdr.split('\n') if l.lower().startswith('retry-after')]
            wait=int(ra[0].split(':')[1]) if ra else 120
            time.sleep(wait+2); continue
        try: j=json.loads(body); msg=f"accepted id {j['level']['id']}" if j.get('accepted') else f"refused: {j.get('reason')}"
        except Exception: msg=body[:100]
        log.write(f"{time.strftime('%F %T')} {label} exit {e} ({m}): {msg}\n"); log.flush(); break
    time.sleep(61)
log.write("done\n")
