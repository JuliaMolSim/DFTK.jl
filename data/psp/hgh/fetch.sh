#!/bin/bash

ELEMENTS=(
    H He
    Li Be B C N O F Ne
    Na Mg Al Si P S Cl Ar
    K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
    Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra Rf Db Sg Bh Hs Mt Ds Rg Cn Uut Fl Uup Lv Uus Uuo
    La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
    Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No Lr
)

URL="https://www.cp2k.org/static/potentials/cp2k/"

for type in pade pbe; do
    for el in ${ELEMENTS[@]}; do
        for q in {1..30}; do
            if [ "$type" == "pade" ]; then
                NAME=$(echo "lda/$el-q$q.hgh" | tr 'A-Z' 'a-z')
            else
                NAME=$(echo "$type/$el-q$q.hgh" | tr 'A-Z' 'a-z')
            fi
            [ -f "$NAME" ] && continue

            if ! wget "$URL/$type/$el-q$q" -O "$NAME"; then
                rm -f "$NAME"
            fi
            sleep 1
        done
    done
done
