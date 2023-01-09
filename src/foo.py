#!/usr/bin/env python3

import argparse
import sys


def main(s):
    print(f"Your input string: {s}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", action="store",
                        help="any string as positional argument")

    args = parser.parse_args()
    sys.exit(main(args.input))
