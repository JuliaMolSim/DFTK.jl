#!/bin/bash
cd "$(dirname "$0")"
rm -f silicon_TB09.log silicon_TB09.abo __ABI_MPIABORTFILE__ *.nc
mpirun -np 4 abinit  silicon_TB09.abi &> silicon_TB09.log
