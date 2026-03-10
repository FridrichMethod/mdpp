#!/bin/bash

cd toppar || exit
rm processed_*
for d in *.itp; do
    cp "$d" processed_"$d"
done
rm processed_forcefield.itp
cd ..
touch cv.dat
while read line; do
    segid=$(echo "$line" | awk '{print $1}')
    resid=$(echo "$line" | awk '{print $2}')
    aname=$(echo "$line" | awk '{print $3}')

    nl toppar/processed_"$segid".itp >toppar/temp.1
    begline=$(grep 'atoms' toppar/temp.1 | awk '{print $1}')
    endline=$(grep 'bonds' toppar/temp.1 | awk '{print $1}')
    if [ -z "$endline" ]; then
        endline=$(grep 'settles' toppar/temp.1 | awk '{print $1}')
    fi

    awk -v var1="$resid" -v var2="$aname" '$4 == var1 && $6 == var2 {print $1}' toppar/temp.1 >>toppar/temp.list

    while read line; do # variable `line` shadowing?
        awk -v var1="$line" -v var2="$begline" -v var3="$endline" '$1 == var1 && $1 > var2 && $1 < var3 {$3=$3"_"}1' toppar/temp.1 >toppar/temp.2
        mv toppar/temp.2 toppar/temp.1
    done <toppar/temp.list
    awk -F" " '{$1=""; print $0}' toppar/temp.1 >toppar/processed_"$segid".itp
    rm toppar/temp.*
done <spt.pdb
cat toppar/forcefield.itp toppar/processed_*.itp >toppar/processed.top
sed -i '/^#/d' topol.top
