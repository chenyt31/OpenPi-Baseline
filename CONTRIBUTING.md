# Contributing to openpi

We welcome contributions, improvements, and modifications. Everyone is welcome to use openpi in accordance to the [license](LICENSE). Contributors are also welcome to submit bug reports, feature requests, and pull requests. We can't promise to approve every pull request, and we are a small team with limited bandwidth to review all requests, but we'll give it our best effort. Specifics are described below.

## Issues and feature requests

If you found a bug or other issue, please first check that the issue was not already reported (use the search bar on Github under Issues). If the issue has not yet been reported, make sure to include this information when filing a Github issue:

- Your OS type and version and the version of Python you are using
- Code that allows us to reproduce your bug, including all dependencies
- Full traceback of any exception
- Any other information that would help us, such as a screenshot

In order for us to address any issue, we must be able to reproduce it, so if you encountered the issue after making modifications to openpi, please reproduce the issue without any other modifications and provide a code snippet that allows us to quickly reproduce the problem on `main`.

If you would like to submit a feature request, please check that the feature request does not already exist, and please provide the following information:

- The motivation for the feature
- A description of the problem you are trying to solve or your use case
- Enough information for us to understand the nature of the request
- Some information for how you intend to use it (this might help us in understanding the motivation!)

We can't promise to support every feature request, but it is helpful to us to know the use cases that you are interested in!

## Submitting a pull request

If you implemented support for a new robot or environment, or some other new feature, we welcome pull requests (PRs) to openpi. We may have limited bandwidth to review PRs in a timely manner and might not be able to offer extensive feedback. We recommend every contribution to consider the following:

- Make sure that your PR has a clear title and description
- Run `pre-commit` (install using `pre-commit install` first)
- Make sure your PR passes all tests
