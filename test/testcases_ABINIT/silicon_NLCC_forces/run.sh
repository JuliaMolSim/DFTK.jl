#!/bin/bash
cd "$(dirname "$0")"
PREFIX="silicon_NLCC_forces"
PSEUDO="Si.psp8"
PSEUDO_FAMILY="pd_nc_sr_lda_standard_0.4.1_psp8"
URL="https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/1bb181334e7298202ceebb4c82e285a9a07ee58f/pseudos/$PSEUDO_FAMILY/$PSEUDO"

if [ ! -f $PSEUDO ]; then
    echo "Downloading pseudopotentials";
    wget $URL
fi

if [ ! -f $PSEUDO ]; then
    echo "Pseudopotential download failed!";
    exit 1
fi

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
