
set -euo pipefail
cd /public_hw/home/cit_yingxinlai/project/nature
mkdir -p slurm_logs
source /public_hw/home/cit_yingxinlai/anaconda3/etc/profile.d/conda.sh
conda activate rppg-h100
export PYTHONPATH=/public_hw/home/cit_yingxinlai/project/nature/.vendor_numpy126:${PYTHONPATH:-}:$PWD

VARIANT=prototype_preserving_slow_rr_ratebin_srcwarm_base
PRETRAIN=runs/exp/prototype_preserving_slow_rr_ratebin_source_warmpretrain/${VARIANT}/${VARIANT}_best.pth
if [[ ! -f "${PRETRAIN}" ]]; then
  echo "Missing pretrained checkpoint: ${PRETRAIN}" >&2
  exit 3
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TARGET=zpu
    MANIFEST_DIR=/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists
    TRAIN_CSV="$MANIFEST_DIR/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_zpu_train_10pct.csv"
    VALID_CSV="$MANIFEST_DIR/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_zpu_valid_10pct.csv"
    TEST_CSV="$MANIFEST_DIR/MultiTask_PPG_PR_HRpleth512_SpO2_72x72_160_clean_zpu_test.csv"
    HR_LOW=45
    HR_HIGH=180
    RR_HIGH=45
    TASKS=hr,rr
    SELECTION=hrrr_mae
    RR_WEIGHT=3.0
    CONTEXT_CLIPS=4
    EVAL_STRIDE=4
    ;;
  1)
    TARGET=kwh
    MANIFEST_DIR=/public_hw/share/cit_ztyu/zhaobo/KWH-rPPG_RAW_160_72x72_Y5F/DataFileLists_MT600
    TRAIN_CSV="$MANIFEST_DIR/KWH_MT600_train_0.0_0.1.csv"
    VALID_CSV="$MANIFEST_DIR/KWH_MT600_valid_0.1_0.2.csv"
    TEST_CSV="$MANIFEST_DIR/KWH_MT600_test_0.8_1.0.csv"
    HR_LOW=78
    HR_HIGH=210
    RR_HIGH=60
    TASKS=hr,rr
    SELECTION=hrrr_mae
    RR_WEIGHT=1.5
    CONTEXT_CLIPS=4
    EVAL_STRIDE=2
    ;;
  2)
    TARGET=szc
    MANIFEST_DIR=/public_hw/share/cit_ztyu/zhaobo/SZC-NICU_unzip/SZC-NICU/DataFileLists_MT
    TRAIN_CSV="$MANIFEST_DIR/SZC_MT_train_0.0_0.1.csv"
    VALID_CSV="$MANIFEST_DIR/SZC_MT_valid_0.0_0.1.csv"
    TEST_CSV="$MANIFEST_DIR/SZC_MT_test_0.1_1.0.csv"
    HR_LOW=78
    HR_HIGH=210
    RR_HIGH=45
    TASKS=hr
    SELECTION=hr_mae
    RR_WEIGHT=0.0
    CONTEXT_CLIPS=1
    EVAL_STRIDE=1
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

MODEL_NAME=${VARIANT}_${TARGET}_fs10_validselect
RUN_DIR=runs/exp/prototype_preserving_slow_rr_ratebin_srcwarm_target_fewshot10_validselect/${MODEL_NAME}
PROVENANCE_DIR=${RUN_DIR}/_provenance
mkdir -p "${PROVENANCE_DIR}"
cp neural_methods/model/SpectroPhys.py "${PROVENANCE_DIR}/SpectroPhys.py"
cp tools/train_multitask_spectrophys.py "${PROVENANCE_DIR}/train_multitask_spectrophys.py"
cp scripts/submit_prototype_preserving_slow_rr_ratebin_srcwarm_target_fewshot10_validselect.sbatch "${PROVENANCE_DIR}/target.sbatch"
sha256sum neural_methods/model/SpectroPhys.py tools/train_multitask_spectrophys.py scripts/submit_prototype_preserving_slow_rr_ratebin_srcwarm_target_fewshot10_validselect.sbatch > "${PROVENANCE_DIR}/sha256sum.txt"

echo "variant=${VARIANT} target=${TARGET} model=${MODEL_NAME} temporal_module=prototype_preserving_slow_rr rate_bin_scalar_mode=blend warmstart=proto_slow_rr_source selection=valid"
python -u tools/train_multitask_spectrophys.py \
  --train-csv "$TRAIN_CSV" \
  --valid-csv "$VALID_CSV" \
  --test-csv "$TEST_CSV" \
  --output-dir runs/exp/prototype_preserving_slow_rr_ratebin_srcwarm_target_fewshot10_validselect \
  --model-name "$MODEL_NAME" \
  --pretrained-checkpoint "$PRETRAIN" \
  --epochs 24 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 4e-5 \
  --weight-decay 1e-4 \
  --frames 160 \
  --context-clips "$CONTEXT_CLIPS" \
  --context-stride 1 \
  --eval-context-stride "$EVAL_STRIDE" \
  --fs 30 \
  --input-normalization channel_mean_center \
  --waveform-head-type gru \
  --temporal-module prototype_preserving_slow_rr \
  --physio-long-context \
  --physio-num-freq-bins 48 \
  --physio-hr-low-bpm "$HR_LOW" \
  --physio-hr-high-bpm "$HR_HIGH" \
  --physio-rr-low-bpm 6 \
  --physio-rr-high-bpm "$RR_HIGH" \
  --vital-representation-adapter \
  --ppg-weight 1.0 \
  --hr-weight 1.0 \
  --rr-weight "$RR_WEIGHT" \
  --hr-from-ppg-weight 0.25 \
  --hr-fft-low-bpm "$HR_LOW" \
  --hr-fft-high-bpm "$HR_HIGH" \
  --hr-fft-nfft 2048 \
  --hr-fft-temperature 10.0 \
  --rate-bin-aux-weight 0.4 \
  --rate-bin-num-bins 96 \
  --rate-bin-hr-sigma-bpm 6.0 \
  --rate-bin-rr-sigma-bpm 1.5 \
  --rate-bin-scalar-mode blend \
  --rate-bin-entropy-weight 0.01 \
  --rate-bin-consistency-weight 0.1 \
  --state-rate-aux-weight 0.5 \
  --state-rate-ppg-consistency-weight 0.15 \
  --state-rate-ppg-rr-weight 0.7 \
  --context-smooth-weight 0.1 \
  --context-feature-smooth-weight 0.5 \
  --context-hr-smooth-weight 1.0 \
  --context-rr-smooth-weight 0.7 \
  --scalar-hr-low-bpm "$HR_LOW" \
  --scalar-hr-high-bpm "$HR_HIGH" \
  --scalar-rr-low-bpm 6 \
  --scalar-rr-high-bpm "$RR_HIGH" \
  --selection-metric "$SELECTION" \
  --train-scalar-tasks "$TASKS" \
  --eval-test-every-epoch \
  --recon-weight 0.02 \
  --freeze-non-scalar \
  --unfreeze-temporal-model \
  --unfreeze-vital-representation-adapter \
  --unfreeze-waveform-head \
  --grad-clip 3.0
