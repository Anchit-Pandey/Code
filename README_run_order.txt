=========================================================================
  CS3T-Lite vs CS3T-UNet  —  Complete Run Order
  5 km/h and 120 km/h  |  L=1 and L=5
=========================================================================

FILE LIST
─────────
  MATLAB (data generation):
    script.m                  — Generate 5 km/h dataset  (original)
    script_120kmh_optionB.m   — Generate 120 km/h dataset (dt=0.5ms)

  Python (core):
    dataset.py                — Data loader  (shared by all scripts)
    model.py                  — CS3T-Lite architecture
    cs3t_unet.py              — CS3T-UNet baseline architecture
    train.py                  — Train CS3T-Lite
    train_baseline.py         — Train CS3T-UNet baseline
    evaluate.py               — Evaluate CS3T-Lite checkpoints
    compare.py                — Side-by-side comparison table
    architecture_summary.py   — Print block shapes + params (no training)

=========================================================================
STEP 1 — Generate datasets  (MATLAB)
=========================================================================

  Run in MATLAB:

  --- 5 km/h ---
  >> run('script.m')

  Output files:
    train_adp.mat       (9000 samples, shape [64,64,2,20,9000])
    test_adp.mat        (1000 samples)
    quadriga_raw_CSI_v2.mat

  Check in console:
    snaps min=20 max=20  ← must see this every environment
    Adjacent corr ≈ 0.9672
    Predicted NMSE ≈ -11.83 dB

  --- 120 km/h (Option B, realistic vehicular) ---
  >> run('script_120kmh_optionB.m')

  Output files:
    train_adp_120B.mat   (9000 samples)
    test_adp_120B.mat    (1000 samples)
    quadriga_raw_CSI_120B.mat

  Check in console:
    snaps min=20 max=20
    Adjacent corr ≈ 0.3717
    Predicted NMSE ≈ +0.99 dB  (correct — harder problem)

=========================================================================
STEP 2 — Verify architecture + parameter counts  (optional but useful)
=========================================================================

  python architecture_summary.py --L 1
  python architecture_summary.py --L 5

  Expected output:
    CS3T-Lite:  11,527,750 params (11.53M)
    CS3T-UNet:  19,608,002 params (19.61M)  ← paper states 19.64M

=========================================================================
STEP 3 — Train CS3T-Lite  (4 runs: 2 speeds × 2 L values)
=========================================================================

  --- 5 km/h ---
  python train.py --L 1 --epochs 400 --bs 32 \
      --train_mat train_adp.mat --test_mat test_adp.mat \
      --results_dir results_5km

  python train.py --L 5 --epochs 400 --bs 64 \
      --train_mat train_adp.mat --test_mat test_adp.mat \
      --results_dir results_5km

  --- 120 km/h ---
  python train.py --L 1 --epochs 400 --bs 32 \
      --train_mat train_adp_120B.mat --test_mat test_adp_120B.mat \
      --results_dir results_120km

  python train.py --L 5 --epochs 400 --bs 64 \
      --train_mat train_adp_120B.mat --test_mat test_adp_120B.mat \
      --results_dir results_120km

  Checkpoints saved to:
    results_5km/checkpoint_L1_best.pt
    results_5km/checkpoint_L5_best.pt
    results_120km/checkpoint_L1_best.pt
    results_120km/checkpoint_L5_best.pt

=========================================================================
STEP 4 — Train CS3T-UNet baseline  (4 runs: 2 speeds × 2 L values)
=========================================================================

  --- 5 km/h ---
  python train_baseline.py --L 1 --epochs 400 --bs 32 \
      --train_mat train_adp.mat --test_mat test_adp.mat \
      --results_dir results_baseline_5km

  python train_baseline.py --L 5 --epochs 400 --bs 32 \
      --train_mat train_adp.mat --test_mat test_adp.mat \
      --results_dir results_baseline_5km

  --- 120 km/h ---
  python train_baseline.py --L 1 --epochs 400 --bs 32 \
      --train_mat train_adp_120B.mat --test_mat test_adp_120B.mat \
      --results_dir results_baseline_120km

  python train_baseline.py --L 5 --epochs 400 --bs 32 \
      --train_mat train_adp_120B.mat --test_mat test_adp_120B.mat \
      --results_dir results_baseline_120km

  Checkpoints saved to:
    results_baseline_5km/baseline_L1_best.pt
    results_baseline_5km/baseline_L5_best.pt
    results_baseline_120km/baseline_L1_best.pt
    results_baseline_120km/baseline_L5_best.pt

=========================================================================
STEP 5 — Evaluate CS3T-Lite  (generates plots)
=========================================================================

  python evaluate.py --results_dir results_5km
  python evaluate.py --results_dir results_120km

=========================================================================
STEP 6 — Full comparison table
=========================================================================

  python compare.py \
      --results_lite       results_5km \
      --results_baseline   results_baseline_5km \
      --test_5km           test_adp.mat \
      --test_120km         test_adp_120B.mat

  NOTE: For 120 km/h comparison, point to 120km results:
  python compare.py \
      --results_lite       results_120km \
      --results_baseline   results_baseline_120km \
      --test_5km           test_adp.mat \
      --test_120km         test_adp_120B.mat

  Expected output table:
  ──────────────────────────────────────────────────────────────────────
  Model                    Params   5km L=1  5km L=5  120km L=1  120km L=5
  ──────────────────────────────────────────────────────────────────────
  CS3T-UNet (baseline)    19.61M   -27.47   -20.58      X dB      X dB
  CS3T-Lite (ours)        11.53M   -48.40   -28.26      X dB      X dB
  Gain (Lite - Baseline)           -20.93    -7.68
  ──────────────────────────────────────────────────────────────────────
  Negative gain = CS3T-Lite is BETTER (lower NMSE)

=========================================================================
TOTAL GPU TIME ESTIMATE  (single A100 / V100)
=========================================================================

  CS3T-Lite  L=1  400 epochs  : ~10 hours
  CS3T-Lite  L=5  400 epochs  : ~10 hours
  CS3T-UNet  L=1  400 epochs  : ~16 hours  (heavier model)
  CS3T-UNet  L=5  400 epochs  : ~16 hours
  × 2 speeds = ~104 hours total

  TIP: Run all 8 jobs in parallel if 8 GPUs available:
    CUDA_VISIBLE_DEVICES=0 python train.py --L 1 ...  &
    CUDA_VISIBLE_DEVICES=1 python train.py --L 5 ...  &
    CUDA_VISIBLE_DEVICES=2 python train_baseline.py --L 1 ...  &
    ...

=========================================================================
DEPENDENCY CHECK
=========================================================================

  Python packages required:
    torch >= 2.0
    numpy
    scipy
    h5py
    matplotlib

  Install:
    pip install torch numpy scipy h5py matplotlib

=========================================================================
QUICK SANITY CHECK  (before full training)
=========================================================================

  # Verify both models load and forward pass correctly:
  python cs3t_unet.py      # should print ~19.61M params ✓
  python model.py          # should print ~11.53M params ✓

  # Verify dataset loads correctly:
  python -c "
  from dataset import QuaDRiGaDataset
  d = QuaDRiGaDataset('train_adp.mat', T=10, L=1, name='test')
  print(f'Dataset OK: {len(d)} windows')
  x, y = d[0]
  print(f'X={x.shape} Y={y.shape}')
  "
