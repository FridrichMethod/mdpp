#!/bin/bash

set -euo pipefail

bash gmx_rmsd.sh
bash gmx_rmsf.sh
bash gmx_gyrate.sh
bash gmx_sasa.sh
bash gmx_energy.sh
bash gmx_dssp.sh
bash gmx_hbond.sh
bash gmx_cluster.sh
