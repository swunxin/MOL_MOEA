#!/usr/bin/env bash
set -euo pipefail

# Run from inside "PlatEMO 4.2". This script cleans QuickVina cache folders
# and PlatEMO MATLAB log files, then starts MATLAB in batch mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATEMO_DIR="$SCRIPT_DIR"
PROJECT_DIR="$(cd "$PLATEMO_DIR/.." && pwd)"
QVINA_DIR="$PROJECT_DIR/QuickVinaTwoGPU"

MATLAB_FUNC="${1:-no_gui_dock_gen}"
LOG_FILE="${2:-matlab_dock_gen.log}"
LOG_PATH="$PLATEMO_DIR/$LOG_FILE"

echo "[INFO] PlatEMO dir: $PLATEMO_DIR"
echo "[INFO] Project dir: $PROJECT_DIR"
echo "[INFO] QuickVina dir: $QVINA_DIR"
echo "[INFO] MATLAB function: $MATLAB_FUNC"
echo "[INFO] MATLAB log: $LOG_PATH"

for d in "$QVINA_DIR/log" "$QVINA_DIR/config" "$QVINA_DIR/ligand_files" "$QVINA_DIR/output"; do
  if [[ -d "$d" ]]; then
    echo "[CLEAN] Removing $d"
    rm -rf "$d"
  fi
  mkdir -p "$d"
done

echo "[CLEAN] Removing PlatEMO MATLAB logs"
rm -f "$PLATEMO_DIR"/matlab*.log

cd "$PLATEMO_DIR"

echo "[RUN] Starting MATLAB batch..."
conda deactivate 2>/dev/null || true
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u QT_PLUGIN_PATH -u GTK_PATH -u GTK_MODULES \
  matlab -batch "$MATLAB_FUNC" > "$LOG_PATH" 2>&1

echo "[DONE] MATLAB finished. Log: $LOG_PATH"
