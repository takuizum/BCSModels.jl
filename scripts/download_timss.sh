#!/usr/bin/env bash
# Download instructions for the TIMSS 2019 Grade-8 mathematics international
# database. The IEA does not allow automated bulk downloads, so this script
# only prints the manual steps.

set -euo pipefail

DATA_DIR="${1:-data/raw/timss2019}"
mkdir -p "$DATA_DIR"

cat <<EOF
TIMSS 2019 international database — manual download steps
==========================================================

1. Open https://timss2019.org/international-database/ in a browser.
2. Accept the data licence (TIMSS data is free for non-commercial research).
3. Download the SPSS bundle for Grade 8 mathematics (file is named
   T19_G8_SPSSData.zip — ~600 MB).
4. Extract it into:
       $DATA_DIR
   so that you have BSA<ctry>M7.sav and BSG<ctry>M7.sav for each country
   code (e.g. BSAUSAM7.sav, BSGUSAM7.sav for the U.S.).

5. Install ReadStatTables.jl into the project environment:
       julia --project=. -e 'using Pkg; Pkg.add("ReadStatTables")'

After step 4 you can run:
       julia --project=. scripts/timss_analysis.jl --country USA --booklet 1

Optional companion files:
  * Item Information Files (IIF) for the booklet → item-code mapping.
  * User Guide for the International Database 2nd Edition (2021):
        https://timss2019.org/international-database/downloads/TIMSS-2019-User-Guide-for-the-International-Database-2nd-Ed.pdf

EOF
