import argparse
import logging
from typing import Any

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from .coder import CSVCoder
from .transform import GetCartesianProduct, ProcessPair

logger = logging.getLogger(__name__)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level.",
    )
    parser.add_argument(
        "--input_1",
        type=str,
        default="input.json",
    )
    parser.add_argument(
        "--input_2",
        type=str,
        default="input.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
    )

    options, reminder = parser.parse_known_args()
    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    transform = MainTransform(**vars(options))
    with beam.Pipeline(options=PipelineOptions(reminder)) as pipeline:
        (pipeline | "MainTransform" >> transform)


class MainTransform(beam.PTransform):  # type: ignore
    def __init__(
        self,
        input_1: str,
        input_2: str,
        output: str,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_1 = input_1
        self.input_2 = input_2
        self.output = output

    def expand(self, pcoll: beam.PCollection) -> None:
        pcoll | beam.Create(list(range(1))) | beam.ParDo(
            GetCartesianProduct(db_name_1=self.input_1, db_name_2=self.input_2)
        ) | beam.Reshuffle() | beam.ParDo(
            ProcessPair(db_name_1=self.input_1, db_name_2=self.input_2)
        ) | beam.io.WriteToText(
            file_path_prefix=self.output,
            file_name_suffix=".csv",
            shard_name_template="",
            num_shards=1,
            coder=CSVCoder(),
        )
        # | beam.ParDo(print)


if __name__ == "__main__":
    run()
