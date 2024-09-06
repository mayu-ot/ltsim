TIMESTAMP=$(date "+%Y%m%d%H%M%S")

INPUT_1=$1
INPUT_2=$2
if [ "${INPUT_1}" = "" ]; then
    echo "Please specify INPUT as the first argument."
    exit;
fi

OUT_DIR="tmp/output"
poetry run python -m comp_pair.main \
    --input_1 "${INPUT_1}" \
    --input_2 "${INPUT_2}" \
    --output "${OUT_DIR}" \
    --temp_location "tmp/temp" \
    ${@:2}
