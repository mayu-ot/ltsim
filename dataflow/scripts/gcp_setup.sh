# TODO: change these lines!
USER="your_user_name"
USER_ID="your_user_id"

NOW=$(date "+%Y%m%d%H%M%S")
RUNNER=${RUNNER:-"DirectRunner"}

# DataflowRunner specific setup
PROJECT=${PROJECT:-"your_project_id"}
REGION=${REGION:-"your_region"}
IMAGE_URI="asia.gcr.io/${PROJECT}/${USER}-${REPOSITORY:-"comp_pair"}:${TAG:-latest}"
GCS_ROOT="gs://${USER}/comp_pair"
