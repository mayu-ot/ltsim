#!/bin/bash

set -eux

source scripts/gcp_setup.sh

JOB_NAME="comp-pair-${NOW}"
JOB_DIR="gs://${USER}/comp_pair"

INPUT_1=$1
INPUT_2=$2
OUT_NAME=$3
if [ "${INPUT_1}" = "" ]; then
    echo "Please specify INPUT as the first argument."
    exit;
fi
OUT_DIR="${JOB_DIR}/${OUT_NAME}"

poetry run python -m comp_pair.main \
    --input_1 "${INPUT_1}" \
    --input_2 "${INPUT_2}" \
    --output="${OUT_DIR}" \
    --runner=DataflowRunner \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --machine_type="n1-highmem-2" \
    --job_name="${JOB_NAME}" \
    --staging_location="${JOB_DIR}/staging" \
    --temp_location="${JOB_DIR}/temp" \
    --experiments use_runner_v2 \
    --sdk_container_image "${IMAGE_URI}" \
    --sdk_location container \
    --labels=owner="${USER_ID}" \
    --max_num_workers 100 \
    --save_main_session \
    ${@:2}
