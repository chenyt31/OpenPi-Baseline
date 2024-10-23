"""This is an example of how to use the openpi package.

To run this example:
> uv run examples/hello.py
"""

import os

from openpi import greeter


def main():
    greeter.greet(os.environ["USER"])


if __name__ == "__main__":
    main()
