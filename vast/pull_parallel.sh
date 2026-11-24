#!/bin/bash
# Pull one remote file over N parallel scp streams (a single stream from the US host crawls at ~50KB/s; 32 streams ~4MB/s).
# Splits on the remote, fetches parts with staggered starts (sshd drops >~10 simultaneous new logins), joins, checks sha256.
#   bash vast/pull_parallel.sh -p PORT root@HOST /workspace/kv2/ko-search-v2/runs/sft_v2/final/model.safetensors runs/sft_v2/final/ 32
set -euo pipefail
[ "$1" = "-p" ] || { echo "usage: $0 -p PORT USER@HOST REMOTE_FILE LOCAL_DIR [N]"; exit 1; }
PORT=$2; HOST=$3; RF=$4; LD=$5; N=${6:-32}
name=$(basename "$RF"); tmp="/tmp/pp_$$"; local_tmp="$LD/.pp_$name"
mkdir -p "$LD" "$local_tmp"
want=$(ssh -p "$PORT" "$HOST" "mkdir -p $tmp && cd $tmp && split -n $N -a 3 '$RF' part_ && sha256sum '$RF' | cut -d' ' -f1" 2>/dev/null | tail -1)
parts=$(ssh -p "$PORT" "$HOST" "ls $tmp" 2>/dev/null | grep '^part_')
for p in $parts; do
  ( for t in 1 2 3; do scp -q -P "$PORT" "$HOST:$tmp/$p" "$local_tmp/" 2>/dev/null && break; sleep 2; done ) &
  sleep 0.3
done
wait
cat "$local_tmp"/part_* > "$LD/$name"
got=$(shasum -a 256 "$LD/$name" | cut -d' ' -f1)
ssh -p "$PORT" "$HOST" "rm -rf $tmp" 2>/dev/null || true
rm -rf "$local_tmp"
if [ "$got" = "$want" ]; then echo "OK $name sha256 $got"; else echo "MISMATCH $name want $want got $got"; exit 2; fi
