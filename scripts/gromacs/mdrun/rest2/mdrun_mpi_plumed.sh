#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production
EQUILIBRATION=step4.3_equilibration
PLUMED_INPUT=cv.dat
TOPPAR_DIR=toppar

# Keep the same temperature ladder as the CHARMM-GUI README example.
REPLICA_LADDER=(303.15 315.40 328.14 341.40 355.19 369.54 384.47 400.00)
NUM_REPLICAS=${#REPLICA_LADDER[@]}

for input in "${PRODUCTION}.mdp" "${EQUILIBRATION}.gro" index.ndx topol.top spt.pdb; do
    if [[ ! -f "${input}" ]]; then
        echo "Required input not found: ${input}" >&2
        exit 1
    fi
done

if [[ ! -d "${TOPPAR_DIR}" ]]; then
    echo "Required directory not found: ${TOPPAR_DIR}" >&2
    exit 1
fi

touch "${PLUMED_INPUT}"

REPLICA_DIRS=()
for replica in $(seq 1 "${NUM_REPLICAS}"); do
    REPLICA_DIRS+=("${replica}")
done

checkpoint_count=0
for replica_dir in "${REPLICA_DIRS[@]}"; do
    if [[ -s "${replica_dir}/${PRODUCTION}.cpt" ]]; then
        ((checkpoint_count += 1))
    fi
done

MDRUN_FLAGS=(
    -v
    -ntomp "${OMP_NUM_THREADS:-1}"
    -pin on
    -multidir "${REPLICA_DIRS[@]}"
    -deffnm "${PRODUCTION}"
    -replex 100
    -plumed "../${PLUMED_INPUT}"
    -hrex
    -dlb no
)

if [[ "${checkpoint_count}" -eq "${NUM_REPLICAS}" ]]; then
    gmx_mpi mdrun -h >/dev/null
    mpirun -np "${NUM_REPLICAS}" gmx_mpi mdrun \
        "${MDRUN_FLAGS[@]}" \
        -cpi "${PRODUCTION}.cpt"
    exit 0
fi

if [[ "${checkpoint_count}" -ne 0 ]]; then
    echo "Found ${checkpoint_count}/${NUM_REPLICAS} replica checkpoints. Refusing mixed restart state." >&2
    echo "Either keep all checkpoints for restart, or remove all replica checkpoints for a fresh start." >&2
    exit 1
fi

bash rest2.sh

base_temperature="${REPLICA_LADDER[0]}"
for replica in "${REPLICA_DIRS[@]}"; do
    mkdir -p "${replica}"
    cp "${PRODUCTION}.mdp" "${replica}/${PRODUCTION}.mdp"

    temperature="${REPLICA_LADDER[$((replica - 1))]}"
    lambda=$(awk -v temp="${temperature}" -v base="${base_temperature}" \
        'BEGIN { printf "%.10f", temp / base }')
    plumed partial_tempering "${lambda}" \
        <"${TOPPAR_DIR}/processed.top" \
        >"${TOPPAR_DIR}/topol${replica}.top"

    sed -e '11i\#include "../toppar/topol.top"' topol.top >"${replica}/topol${replica}.top"
    sed -i "s/topol\.top/topol${replica}.top/" "${replica}/topol${replica}.top"
done

for replica in "${REPLICA_DIRS[@]}"; do
    GROMPP_FLAGS=(
        -f "${replica}/${PRODUCTION}.mdp"
        -o "${replica}/${PRODUCTION}.tpr"
        -c "${EQUILIBRATION}.gro"
        -p "${replica}/topol${replica}.top"
        -n index.ndx
    )

    if [[ -s "${EQUILIBRATION}.cpt" ]]; then
        GROMPP_FLAGS+=(-t "${EQUILIBRATION}.cpt")
    fi

    gmx grompp "${GROMPP_FLAGS[@]}"
done

mpirun -np "${NUM_REPLICAS}" gmx_mpi mdrun "${MDRUN_FLAGS[@]}"
