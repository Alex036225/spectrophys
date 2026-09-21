
set -euo pipefail
cd /public_hw/home/cit_yingxinlai/project/nature
source /public_hw/home/cit_yingxinlai/anaconda3/etc/profile.d/conda.sh
conda activate rppg-h100
export PYTHONPATH=/public_hw/home/cit_yingxinlai/project/nature/.vendor_numpy126:${PYTHONPATH:-}:$PWD

MODELS=(physnet deepphys physformer rhythmformer bigsmall efficientphys multiphysnet ours)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
D=/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists
TRAIN="$D/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_true4source_train.csv"
VALID="$D/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_true4source_valid.csv"
OUT=runs/exp/nmi_true4source_pretrain

if [[ "$MODEL" != ours ]]; then
  python -u tools/train_kwh_standard_waveform.py \
    --model "$MODEL" --train-csv "$TRAIN" --valid-csv "$VALID" --test-csv "$VALID" --output-dir "$OUT" \
    --frames 160 --epochs 16 --batch-size 2 --lr 8e-5 --weight-decay 1e-4 --num-workers 4 \
    --fs 30 --window-seconds 30 --hr-low-bpm 45 --hr-high-bpm 180 --rr-low-bpm 6 --rr-high-bpm 45 --seed 20260731
else
  python -u tools/train_multitask_phasenet.py \
    --train-csv "$TRAIN" --valid-csv "$VALID" --output-dir "$OUT" --model-name ours \
    --epochs 16 --batch-size 4 --num-workers 4 --seed 20260731 --lr 8e-5 --weight-decay 1e-4 \
    --frames 160 --fs 30 --input-normalization channel_mean_center --waveform-head-type gru \
    --temporal-module subharmonic_cardioresp_memory --physio-long-context --physio-num-freq-bins 48 \
    --physio-hr-low-bpm 45 --physio-hr-high-bpm 180 --physio-rr-low-bpm 6 --physio-rr-high-bpm 45 \
    --vital-representation-adapter --scalar-head-type residual_spectral_rate \
    --scalar-hr-low-bpm 45 --scalar-hr-high-bpm 180 --scalar-rr-low-bpm 6 --scalar-rr-high-bpm 45 \
    --ppg-weight 1.0 --pr-weight 0.05 --hr-weight 0.5 --rr-weight 0.2 --spo2-weight 0.02 \
    --hr-from-ppg-weight 0.15 --hr-fft-low-bpm 45 --hr-fft-high-bpm 180 --hr-fft-nfft 2048 \
    --hr-fft-temperature 10.0 --selection-metric loss --recon-weight 0.02 --weighted-sampler \
    --sampler-dataset-balance --sampler-hr-balance --sampler-hr-bin-width 8.0 \
    --sampler-num-samples-multiplier 0.75 --grad-clip 3.0
fi
