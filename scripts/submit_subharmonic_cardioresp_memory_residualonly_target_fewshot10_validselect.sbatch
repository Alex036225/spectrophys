
set -euo pipefail
cd /public_hw/home/cit_yingxinlai/project/nature
mkdir -p slurm_logs
source /public_hw/home/cit_yingxinlai/anaconda3/etc/profile.d/conda.sh
conda activate rppg-h100
export PYTHONPATH=/public_hw/home/cit_yingxinlai/project/nature/.vendor_numpy126:${PYTHONPATH:-}:$PWD

VARIANT=subharmonic_cardioresp_memory_residualonly_base
PRETRAIN=runs/exp/complex_proto_memory_source_pretrain/complex_proto_memory_base/complex_proto_memory_base_best.pth
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
    TASKS=hr,rr
    SELECTION=hrrr_mae
    RR_WEIGHT=2.5
    CONTEXT_CLIPS=4
    ;;
  1)
    TARGET=kwh
    MANIFEST_DIR=/public_hw/share/cit_ztyu/zhaobo/KWH-rPPG_RAW_160_72x72_Y5F/DataFileLists_MT600
    TRAIN_CSV="$MANIFEST_DIR/KWH_MT600_train_0.0_0.1.csv"
    VALID_CSV="$MANIFEST_DIR/KWH_MT600_valid_0.1_0.2.csv"
    TEST_CSV="$MANIFEST_DIR/KWH_MT600_test_0.8_1.0.csv"
    HR_LOW=78
    HR_HIGH=210
    TASKS=hr,rr
    SELECTION=hrrr_mae
    RR_WEIGHT=1.0
    CONTEXT_CLIPS=1
    ;;
  2)
    TARGET=szc
    MANIFEST_DIR=/public_hw/share/cit_ztyu/zhaobo/SZC-NICU_unzip/SZC-NICU/DataFileLists_MT
    TRAIN_CSV="$MANIFEST_DIR/SZC_MT_train_0.0_0.1.csv"
    VALID_CSV="$MANIFEST_DIR/SZC_MT_valid_0.0_0.1.csv"
    TEST_CSV="$MANIFEST_DIR/SZC_MT_test_0.1_1.0.csv"
    HR_LOW=78
    HR_HIGH=210
    TASKS=hr
    SELECTION=hr_mae
    RR_WEIGHT=0.0
    CONTEXT_CLIPS=1
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

MODEL_NAME=${VARIANT}_${TARGET}_fs10_validselect
OUT_ROOT=runs/exp/subharmonic_cardioresp_memory_residualonly_target_fewshot10_validselect
RUN_DIR=${OUT_ROOT}/${MODEL_NAME}
PROVENANCE_DIR=${RUN_DIR}/_provenance
mkdir -p "${PROVENANCE_DIR}"
cp neural_methods/model/SpectroPhys.py "${PROVENANCE_DIR}/SpectroPhys.py"
cp tools/train_multitask_spectrophys.py "${PROVENANCE_DIR}/train_multitask_spectrophys.py"
cp scripts/submit_subharmonic_cardioresp_memory_residualonly_target_fewshot10_validselect.sbatch "${PROVENANCE_DIR}/target.sbatch"
sha256sum neural_methods/model/SpectroPhys.py tools/train_multitask_spectrophys.py scripts/submit_subharmonic_cardioresp_memory_residualonly_target_fewshot10_validselect.sbatch > "${PROVENANCE_DIR}/sha256sum.txt"

echo "variant=${VARIANT} target=${TARGET} model=${MODEL_NAME} temporal_module=subharmonic_cardioresp_memory scalar_head_type=residual_spectral_rate train_new_subharmonic_only=1 selection=valid"
python -u tools/train_multitask_spectrophys.py \
  --train-csv "$TRAIN_CSV" \
  --valid-csv "$VALID_CSV" \
  --test-csv "$TEST_CSV" \
  --output-dir "$OUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --pretrained-checkpoint "$PRETRAIN" \
  --epochs 24 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 8e-5 \
  --weight-decay 1e-4 \
  --frames 160 \
  --context-clips "$CONTEXT_CLIPS" \
  --context-stride 1 \
  --eval-context-stride "$CONTEXT_CLIPS" \
  --fs 30 \
  --input-normalization channel_mean_center \
  --waveform-head-type gru \
  --temporal-module subharmonic_cardioresp_memory \
  --physio-long-context \
  --physio-num-freq-bins 48 \
  --physio-hr-low-bpm "$HR_LOW" \
  --physio-hr-high-bpm "$HR_HIGH" \
  --physio-rr-low-bpm 6 \
  --physio-rr-high-bpm 45 \
  --vital-representation-adapter \
  --scalar-head-type residual_spectral_rate \
  --scalar-hr-low-bpm "$HR_LOW" \
  --scalar-hr-high-bpm "$HR_HIGH" \
  --scalar-rr-low-bpm 6 \
  --scalar-rr-high-bpm 45 \
  --ppg-weight 1.0 \
  --hr-weight 1.0 \
  --rr-weight "$RR_WEIGHT" \
  --hr-from-ppg-weight 0.25 \
  --hr-fft-low-bpm "$HR_LOW" \
  --hr-fft-high-bpm "$HR_HIGH" \
  --hr-fft-nfft 2048 \
  --hr-fft-temperature 10.0 \
  --selection-metric "$SELECTION" \
  --train-scalar-tasks "$TASKS" \
  --eval-test-every-epoch \
  --recon-weight 0.02 \
  --freeze-non-scalar \
  --freeze-scalar-heads \
  --unfreeze-parameter-substrings temporal_model.hr_subharmonic_embed,temporal_model.subharmonic_to_rr,temporal_model.rr_to_carrier,temporal_model.carrier_modulation,temporal_model.subharmonic_ffn,temporal_model.subharmonic_rr_gate,temporal_model.carrier_mod_gate \
  --unfreeze-waveform-head \
  --grad-clip 3.0
