
set -euo pipefail
cd /public_hw/home/cit_yingxinlai/project/nature
source /public_hw/home/cit_yingxinlai/anaconda3/etc/profile.d/conda.sh
conda activate rppg-h100
export PYTHONPATH=/public_hw/home/cit_yingxinlai/project/nature/.vendor_numpy126:${PYTHONPATH:-}:$PWD

MODELS=(physnet deepphys physformer rhythmformer bigsmall efficientphys multiphysnet ours)
TARGETS=(zpu kwh szc)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
TARGET=${TARGETS[$((SLURM_ARRAY_TASK_ID % 3))]}
case "$TARGET" in
  zpu)
    D=/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists
    TRAIN="$D/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_zpu_train_10pct.csv"
    VALID="$D/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_zpu_valid_10pct.csv"
    TEST="$D/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_zpu_test.csv"
    HR_LOW=45; HR_HIGH=180; TASKS=hr,rr,spo2; SELECTION=hrrr_mae; RR_WEIGHT=2.5; SPO2_WEIGHT=0.02; CONTEXT=4 ;;
  kwh)
    D=/public_hw/share/cit_ztyu/zhaobo/KWH-rPPG_RAW_160_72x72_Y5F/DataFileLists_MT600
    TRAIN="$D/KWH_MT600_train_0.0_0.1.csv"; VALID="$D/KWH_MT600_valid_0.1_0.2.csv"; TEST="$D/KWH_MT600_test_0.8_1.0.csv"
    HR_LOW=78; HR_HIGH=210; TASKS=hr,rr; SELECTION=hrrr_mae; RR_WEIGHT=1.0; SPO2_WEIGHT=0.0; CONTEXT=1 ;;
  szc)
    D=/public_hw/share/cit_ztyu/zhaobo/SZC-NICU_unzip/SZC-NICU/DataFileLists_MT
    TRAIN="$D/SZC_MT_clean_fs10_train.csv"; VALID="$D/SZC_MT_clean_fs10_valid.csv"; TEST="$D/SZC_MT_clean_fs10_test.csv"
    HR_LOW=78; HR_HIGH=210; TASKS=hr; SELECTION=hr_mae; RR_WEIGHT=0.0; SPO2_WEIGHT=0.0; CONTEXT=1 ;;
esac
OUT=runs/exp/nmi_true4source_target

if [[ "$MODEL" != ours ]]; then
  CKPT="runs/exp/nmi_true4source_pretrain/${MODEL}/${MODEL}_best.pth"
  python -u tools/train_kwh_standard_waveform.py \
    --model "$MODEL" --train-csv "$TRAIN" --valid-csv "$VALID" --test-csv "$TEST" \
    --output-dir "$OUT/${TARGET}" --init-checkpoint "$CKPT" --frames 160 --epochs 10 --batch-size 1 \
    --lr 1e-4 --weight-decay 1e-4 --num-workers 4 --fs 30 --window-seconds 30 \
    --hr-low-bpm "$HR_LOW" --hr-high-bpm "$HR_HIGH" --rr-low-bpm 6 --rr-high-bpm 45 --seed 20260731
else
  CKPT=runs/exp/nmi_true4source_pretrain/ours/ours_best.pth
  NAME="ours_${TARGET}"
  python -u tools/train_multitask_spectrophys.py \
    --train-csv "$TRAIN" --valid-csv "$VALID" --test-csv "$TEST" --output-dir "$OUT/ours" \
    --model-name "$NAME" --pretrained-checkpoint "$CKPT" --epochs 24 --batch-size 1 --num-workers 4 \
    --seed 20260731 --lr 6e-5 --weight-decay 1e-4 --frames 160 --context-clips "$CONTEXT" \
    --context-stride 1 --eval-context-stride "$CONTEXT" --fs 30 --input-normalization channel_mean_center \
    --waveform-head-type gru --temporal-module subharmonic_cardioresp_memory --physio-long-context \
    --physio-num-freq-bins 48 --physio-hr-low-bpm "$HR_LOW" --physio-hr-high-bpm "$HR_HIGH" \
    --physio-rr-low-bpm 6 --physio-rr-high-bpm 45 --vital-representation-adapter \
    --scalar-head-type residual_spectral_rate --scalar-hr-low-bpm "$HR_LOW" --scalar-hr-high-bpm "$HR_HIGH" \
    --scalar-rr-low-bpm 6 --scalar-rr-high-bpm 45 --ppg-weight 1.0 --hr-weight 1.0 \
    --rr-weight "$RR_WEIGHT" --spo2-weight "$SPO2_WEIGHT" --hr-from-ppg-weight 0.25 \
    --hr-fft-low-bpm "$HR_LOW" --hr-fft-high-bpm "$HR_HIGH" --hr-fft-nfft 2048 \
    --hr-fft-temperature 10.0 --selection-metric "$SELECTION" --train-scalar-tasks "$TASKS" \
    --recon-weight 0.02 --freeze-non-scalar --unfreeze-waveform-head --grad-clip 3.0
fi
