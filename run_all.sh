#!/usr/bin/env bash
# Regenerate every result, table and figure in the replication package.
# Runs sequentially; logs go to logs/<step>.log and logs/PROGRESS records
# what finished. Takes roughly 2h30 on two cores.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"
PROGRESS="$LOGS/PROGRESS"
: > "$PROGRESS"

run () {  # run <name> <dir> <command...>
  local name="$1"; shift
  local dir="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $name" | tee -a "$PROGRESS"
  ( cd "$dir" && "$@" ) > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "[$(date -u +%H:%M:%S)] END   $name rc=$rc" | tee -a "$PROGRESS"
}

FC="$ROOT/forecast_codes"
NC="$ROOT/nowcast_codes"

# ---------------- forecasts ----------------
for T in PCEPILFE CPIAUCSL; do
  run "fc_llama70B_$T"   "$FC" python3 -u new_codes_llama70B.py   --target "$T"
  run "fc_sentiments_$T" "$FC" python3 -u new_codes_sentiments.py --target "$T"
  run "fc_main_$T"       "$FC" python3 -u new_codes.py            --target "$T"
  run "fc_check_$T"      "$FC" python3 -u check_results_forecast.py --target "$T" \
        --export --mad --mae --no-figures
done

# ---------------- nowcasts ----------------
# CPI is released mid-month, so only the +5/+10/+14 cutoffs keep the Reddit
# information strictly prior to the release. PCE is released later and also
# allows +22. Run the cutoffs of a target in one go: the scripts reproduce the
# notebooks' carry-over of `ma_window` across cutoffs (see --ma-init).
run "nc_ft_PCEPILFE"    "$NC" python3 -u new_codes_real_time.py \
      --target PCEPILFE --cutoffs 5 10 14 22 --save
run "nc_llama_PCEPILFE" "$NC" python3 -u new_codes_real_time_llama70B.py \
      --target PCEPILFE --cutoffs 5 10 14 22 --save

run "nc_ft_CPIAUCSL"    "$NC" python3 -u new_codes_real_time.py \
      --target CPIAUCSL --cutoffs 5 10 14 --save
run "nc_llama_CPIAUCSL" "$NC" python3 -u new_codes_real_time_llama70B.py \
      --target CPIAUCSL --cutoffs 5 10 14 --save

# comparison with the Cleveland Fed nowcast, every cutoff (also draws Figures 9-10)
run "nc_fed"            "$NC" python3 -u nowcast_charts.py

# ---------------- tables and figures of the main text ----------------
run "paper_objects" "$ROOT" python3 -u paper_objects.py --no-show

echo "[$(date -u +%H:%M:%S)] ALL DONE" | tee -a "$PROGRESS"
echo
echo "Tables and figures: see $LOGS/paper_objects.log and $ROOT/figures_paper/"
