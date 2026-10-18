#!/usr/bin/env python3
"""Reproduce every table in the paper. Equivalent to `python -m halo all`."""

import sys

from halo.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["all"] + sys.argv[1:]))
