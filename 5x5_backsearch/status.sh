#!/bin/bash
# One-screen status of the exhaustive campaigns: finished monolithic runs in
# results/*.log, then every campaign directory's progress.
cd "$(dirname "$0")"
echo "== finished single runs (results/*.log)"
for f in results/*.log; do
  [ -f "$f" ] || continue
  if grep -q "^--- Exit" "$f"; then
    printf "  %-28s %s\n" "$(basename "$f" .log)" \
      "$(grep -A7 '^--- Exit' "$f" | grep 'Exit\|elapsed\|states checked\|best depth' | tr -s ' ' | sed 's/^ //' | tr '\n' ' ')"
  else
    printf "  %-28s running/incomplete\n" "$(basename "$f" .log)"
  fi
done
echo
echo "== campaigns"
for d in results/camp_*/; do
  [ -f "$d/jobs.tsv" ] || continue
  python3 campaign.py --out "${d%/}" --status 2>/dev/null | sed 's/^/  /'
  w=$(cat "$d/workers" 2>/dev/null); [ -n "$w" ] && echo "     workers file: $w"
done
echo
echo "== processes"
pgrep -fl "campaign.py|backsearch_worker" | cut -c1-110 | sed 's/^/  /'
