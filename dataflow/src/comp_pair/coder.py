import csv
import io
from typing import Any, NamedTuple, Union

import apache_beam as beam


class CSVCoder(beam.coders.Coder):  # type: ignore
    """Converts namedtuple rows to CSV row."""

    def encode(self, row: Union[dict[str, Any], NamedTuple]) -> bytes:
        with io.StringIO(newline="") as f:
            writer = csv.writer(f, lineterminator="")
            writer.writerow(list(row.values() if isinstance(row, dict) else row))
            return f.getvalue().encode("utf-8", "ignore")

    def decode(self, value: bytes) -> list[Any]:
        with io.StringIO(value.decode("utf-8", "ignore")) as f:
            reader = csv.reader(f)
            return next(iter(reader))
