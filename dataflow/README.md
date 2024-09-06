# Overview
Large-scale pairwise comparison on DataFlow

## Development

### Prerequisites

| Software                | Install                                        |
|-------------------------|------------------------------------------------|
| [Python 3.9.17][python] | `pyenv install 3.9.17 && pyenv local 3.9.17` |
| [Poetry 1.5.1][poetry]  | `curl -sSL https://install.python-poetry.org \| python3 -` |

[python]: https://www.python.org/downloads/release/python-3917/
[poetry]: https://python-poetry.org/

### Setup
Note: make sure to make a clean new environment, since Apache Beam is very strict with version mismatch between local and dataflow sides.
```bash
python -m venv .venv
poetry env use .venv/bin/python
poetry install --with dev
```

### Run
EMD is computed for each pair of layouts in the input files. The input files are assumed to be in the format of layout files we provide. The output is a list of [i, j, emd]. i and j are the indices of the layouts in the input files and emd is the EMD between the two layouts.

```bash
# run on local
bash scripts/run_on_local.sh input_1.json input_2.json
```

```bash
# register currecnt code and requirements so that they are accessible from GCP
bash scripts/build_and_push_docker_image.sh
# run on Dataflow
bash scripts/run_on_gcp.sh gs://path/to/input_1.json gs://path/to/input_2.json gs://path/to/output
```
