# layout-eval

# Setup
We use [poetry](https://python-poetry.org/docs/) to manage dependencies. Install poetry and run the following command to install dependencies.
```
poetry install
```

# Data preparation

Download the pre-processed dataset and generated layouts by running the following command.

```
wget https://github.com/mayu-ot/layout-eval/releases/download/v1.0.0/data.zip
unzip data.zip
```

The data directory should look like this:
```
data
├── datasets # post-processed datasets
│   ├── rico25
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   └── publaynet
├──fid_feat # pre-extracted features for FID evaluation
├── results_conditional # generated layouts for conditional layout generation
│   ├── publaynet
│   └── rico
└── results_conditional # generated layouts for unconditional layout generation
    ├── publaynet
    └── rico
        ├── partial # generated layouts for layout completion
        └── c # generated layouts for label-conditioned layout generation
            ├── bart
            ├── ...
            └──vqdiffusion
```

## FID feature extraction
1. Download the [LayoutDM resources](https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip) and copy `download/fid_weights/FIDNetV3/rico25-max25/model_best.pth.tar` to $FID_WEIGHT_FILE.
2. Run the following command to extract layout features to evaluate FID on RICO dataset.

```
python src/experiments/feature_extraction.py \
  --dataset_type rico25 \
  --input_dataset_json $DATASET_JSON \
  --output_feat_file $OUTPUT_FILE_NAME \
  --fid_weight_file $FID_WEIGHT_FILE
```

# Evaluate conditional layout generation
Download generated layouts in `./data` following the [instruction](#data-preparation).
Run the script to get evaluation results on RICO. The results are saved in `data/results/eval_conditional/rico/result.csv`
```
poetry run python src/experiments/eval_conditional.py rico
```

# Evaluate unconditional layout generation
Download generated layouts in `./data` following the [instruction](#data-preparation).
Run the script to get evaluation results on RICO. The results are saved in $RESULT_FILE.
```
poetry run python src/experiments/eval_unconditional.py rico $RESULT_FILE
```


# Demo
To run an iteractive app to try the evaluation metrics, run the following command.
```
streamlit run src/app/measure_explore.py 
```