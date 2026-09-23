#!/bin/bash
module load espresso/7.5-libxc-7.0.0-cpu
export OMP_NUM_THREADS=1
exec pw.x "$@"
