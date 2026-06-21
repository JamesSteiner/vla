#!/usr/bin/env python
"""Get LIBERO into LeRobotDataset format.

Prefer an already-converted LIBERO dataset on the HF hub if one exists for your
suite. Otherwise convert LIBERO's HDF5 demos with LeRobot's dataset tooling
(LeRobotDataset.create / add_frame, or a community conversion script).

Stub — fill in once you've picked the suite (start with one, e.g. libero_spatial).
"""

from __future__ import annotations


def main():
    raise NotImplementedError(
        "Either point vla.data.libero.LIBERO_REPO_ID at a hub dataset, or convert HDF5 here."
    )


if __name__ == "__main__":
    main()
