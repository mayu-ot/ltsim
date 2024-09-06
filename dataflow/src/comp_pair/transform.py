import logging
from itertools import product
from typing import Any, Iterable, Iterator
import numpy as np

# import pickle
import json

from apache_beam.io.filesystems import FileSystems

import apache_beam as beam
from .ltsim import (
    provide_cost_mat_and_dual_var,
    compute_lt_cost_between_layout,
)

logger = logging.getLogger(__name__)


def load_db(db_name: str):
    with FileSystems.open(db_name, "r") as f:
        db = json.load(f)
        if "annotations" in db:
            return list(db["annotations"].values())
        elif "results" in db:
            return list(db["results"].values())


class GetCartesianProduct(beam.DoFn):  # type: ignore
    """
    Load a db, get its size, and return a cartesian product of indexes.
    """

    def __init__(
        self, db_name_1: str, db_name_2: str, partition_size: int = 10, **kwargs: Any
    ) -> None:
        super().__init__()
        self.partition_size = partition_size
        self.db_name_1 = db_name_1
        self.db_name_2 = db_name_2
        self.db_length_1 = 0  # note: this should be set in setup
        self.db_length_2 = 0

    def setup(self) -> None:
        # note: some common load function assumes real file path (e.g., np.load).
        # when loading does not work, change the data type if possible.
        db = load_db(self.db_name_1)
        self.db_length_1 = len(db)
        db = load_db(self.db_name_2)
        self.db_length_2 = len(db)

    def process(self, *args, **kwargs) -> Iterable:  # type: ignore
        """
        Note: currently assuming this part is only called once (load a db and iterate over every pair).
        """
        indexes_1 = list(range(self.db_length_1))
        indexes_2 = list(range(self.db_length_2))

        for i, j in product(indexes_1, indexes_2):
            yield (i, j)


def func(x: Any, y: Any) -> float:
    """
    Take dense feature embeddings and return an inner product as a similarity score.
    Currently, inputs are numpy arrays but it can be changed to other types,
    as long as the function returns a single float value
    """
    return float(np.inner(x, y).item())


def emd(x: Any, y: Any) -> float:
    lt_cost_fnc = lambda x, y: compute_lt_cost_between_layout(
        x, y, provide_cost_mat_and_dual_var
    )
    return lt_cost_fnc(x, y)


class ProcessPair(beam.DoFn):  # type: ignore
    def __init__(
        self,
        db_name_1: str,
        db_name_2: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.db_name_1 = db_name_1
        self.db_name_2 = db_name_2

    def setup(self) -> None:
        self.db_1 = load_db(self.db_name_1)
        self.db_2 = load_db(self.db_name_2)

    def process(self, indexes) -> Iterable:  # type: ignore
        i, j = indexes
        val = emd(self.db_1[i], self.db_2[j])
        yield (i, j, val)
