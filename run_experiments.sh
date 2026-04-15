#!/bin/bash
#SBATCH --job-name=trustworthy-ai
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # <-- adjust to your cluster's GPU partition
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# ── Configuration ──────────────────────────────────────────
MODEL_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"
MODEL_BASE="Qwen/Qwen2.5-7B"
STEERING_LAYER=8
INVERT_LAYER=4
OUTDIR="results"

# ── Environment ────────────────────────────────────────────
# Uncomment / adjust for your cluster's module system:
# module load cuda/12.1
# module load anaconda3
# source activate myenv

set -euo pipefail
mkdir -p "$OUTDIR"

echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Model (steering/GCG): $MODEL_INSTRUCT"
echo "  Model (inversion):    $MODEL_BASE"
echo "  Steering layer:       $STEERING_LAYER"
echo "  Inversion layer:      $INVERT_LAYER"
echo "============================================"

# ── 1. GCG activation-matching attack (suffix + replacement) ──
echo ""
echo ">>> Running GCG experiments..."
python main.py \
    --model "$MODEL_INSTRUCT" \
    --layer "$STEERING_LAYER" \
    --mode both \
    --outdir "$OUTDIR"

# ── 2. Token probe (hidden state → token inversion) ──
echo ""
echo ">>> Training token probe..."
python invert.py \
    --model "$MODEL_BASE" \
    --layer "$INVERT_LAYER" \
    --task token \
    --outdir "$OUTDIR"

# ── 3. Sequence inverter (hidden state → full sequence) ──
echo ""
echo ">>> Training sequence inverter..."
python invert.py \
    --model "$MODEL_BASE" \
    --layer "$INVERT_LAYER" \
    --task sequence \
    --outdir "$OUTDIR"

echo ""
echo ">>> All experiments complete. Results in $OUTDIR/"
ls -lh "$OUTDIR"
