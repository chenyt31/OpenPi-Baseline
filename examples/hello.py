"""This is an example of how to use the openpi package.

To run this example:
> uv run examples/hello.py
"""

import openpi_client

from openpi import greeter


def main():
    greeter.greet(openpi_client.__version__)


if __name__ == "__main__":
    main()
