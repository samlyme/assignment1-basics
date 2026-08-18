from collections.abc import Iterable
import math
import os
from typing import Any, TypedDict
from collections.abc import Callable
import typing

from torch import Tensor
import torch
import numpy as np
import numpy.typing as npt

import argparse

from cs336_basics.utils import get_batch


def main():
    parser = argparse.ArgumentParser(description="Read a file as raw bytes.")
    parser.add_argument(
        "filename",
        type=str,
        help="Path to tokenized training data",
    )
    parser.add_argument("batch_size", type=int)
    parser.add_argument("context_length", type=int)
    parser.add_argument("device", type=str)

    # TODO: implement model config arg
    parser.add_argument("checkpoint", type=str)

    args = parser.parse_args()

    dataset = np.memmap(args.filename, dtype=np.uint16)

    for iteration in range(100):
        batch = get_batch(
            dataset, args.batch_size, args.context_length, args.device
        )


if __name__ == "__main__":
    main()
