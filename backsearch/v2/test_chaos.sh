#!/bin/bash
# Integration chaos test (v2/DESIGN.md §5.2; review M90, M91, M96, T10): real server, real client,
# real worker; a client SIGKILLed while it holds leases, a client stopped with SIGINT mid-window, a
# server restart; the clients must exit by themselves when the campaign is complete, the fresh audit
# must be clean and exact, and the level union over the ACCEPTED runs must equal the monolithic run.
# The driver is test_chaos.py (see its docstring for the scenario and every assertion).
#
#   bash test_chaos.sh WORKER_BINARY [SERVER_DIR] [test_chaos.py options, e.g. --config 6x6h0b1]
#
# Run it under the CPU guard with two slots: a server, two clients and their workers run at once.
set -u
WORKER=${1:?worker binary}; shift
SERVER_DIR=/Users/george/PathologyRecords/server
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then SERVER_DIR=$1; shift; fi
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/test_chaos.py" "$WORKER" --server-dir "$SERVER_DIR" "$@"
