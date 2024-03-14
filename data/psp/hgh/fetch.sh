#!/bin/bash

URL="https://raw.githubusercontent.com/cp2k/cp2k-data/master/potentials/Goedecker/cp2k"

for xcf in pade pbe; do
   element=($(wget -nv ${URL}/GTH_POTENTIALS -O - | grep -i "${xcf}-q" | awk '{print $1}'))
   zpseudo=($(wget -nv ${URL}/GTH_POTENTIALS -O - | grep -i "${xcf}-q" | awk '{print $2}' | cut -d- -f3))
   for ((i=0; i < ${#element[@]}; i++)); do
      [[ ${zpseudo[i]} =~ [^q[:digit:]] ]] && continue
      ppname="${element[i]}-${zpseudo[i]}"
      remote="${xcf}/${ppname}"
      target=$(echo ${remote/pade/lda}.hgh | tr 'A-Z' 'a-z')
      wget -nv ${URL}/${remote} -O ${target}
   done
done
