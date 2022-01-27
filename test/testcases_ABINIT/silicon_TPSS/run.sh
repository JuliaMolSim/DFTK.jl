#!/bin/bash
cd "$(dirname "$0")"
PREFIX="silicon_TPSS"

rm -f ${PREFIX}.log ${PREFIX}.abo __ABI_MPIABORTFILE__ *.nc
mpirun -np 4 abinit  ${PREFIX}.abi &> ${PREFIX}.log

FILES=(
	${PREFIX}o_DEN.nc
	${PREFIX}o_EBANDS.agr
	${PREFIX}o_EIG
	${PREFIX}o_EIG.nc
	${PREFIX}o_KDEN.nc
	${PREFIX}o_OUT.nc
	${PREFIX}o_WFK.nc
	${PREFIX}o_DDB
	__ABI_MPIABORTFILE__
)
rm -f ${FILES[@]}
