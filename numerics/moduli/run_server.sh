#!/bin/bash
# Server driver: one BLAS/OpenMP thread per worker process (the eigensolver is
# serial; without this every worker spawns a full-size BLAS pool), logs in /root/logs.
#   bash numerics/moduli/run_server.sh suite|fullquad|kernel [workers]
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python3}
cd "$(dirname "$0")"
mkdir -p /root/logs
case "$1" in
  suite)    "$PY" solve_moduli.py suite "${2:-14}" ;;
  fullquad) "$PY" solve_moduli.py fullquad ;;
  kernel)   "$PY" kernel.py ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
