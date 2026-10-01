# Code of Conduct

## Our Standards

HeCBench is a collaborative, academic benchmark suite. Many of its benchmarks
are ported or adapted from other open-source projects, and it spans multiple
hardware vendors (e.g. NVIDIA, AMD, Intel) and a growing set of programming
systems. These currently include models such as CUDA, HIP, SYCL/DPC++, and
OpenMP target offloading, and are expected to expand over time to additional
languages and frameworks (for example Fortran, Kokkos, Mojo, Python/Triton,
Rust/cubeCL, and Julia/JACC). Any programming system in or added to the suite is
covered by this Code of Conduct; the models named here are examples, not an
exhaustive list. Our standards reflect this research-oriented, multi-vendor,
multi-contributor, and multi-language nature.

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other contributors
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Properly attributing and crediting the original authors of any code, dataset,
  algorithm, or benchmark that is ported or adapted into HeCBench, and honoring
  the associated licenses
* Remaining vendor- and platform-neutral: comparing hardware, vendors, and
  programming models on technical merit and reporting results honestly and
  reproducibly

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or advances of
  any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Publishing others' private information, such as a physical or email address,
  without their explicit permission
* Plagiarism, misrepresenting authorship, or removing/omitting required license
  headers, copyright notices, or attribution of upstream sources
* Fabricating, falsifying, cherry-picking, or otherwise misrepresenting
  benchmark results, or intentionally biasing benchmarks to favor or disparage a
  particular vendor, device, or programming model
* Other conduct which could reasonably be considered inappropriate in a
  professional or academic setting

## Research and Benchmarking Integrity

Because HeCBench is used to study performance, portability, and productivity, the
integrity of its code and results is essential. Contributors are expected to:

* **Attribution and licensing.** When adding a benchmark that is ported or
  adapted from an external project, retain the original license and copyright
  notices, comply with that license, and link to the upstream source (as is done
  throughout the `README.md`). Do not submit code you are not permitted to
  redistribute.
* **Honest results.** Report performance and correctness results truthfully and,
  where possible, reproducibly. Do not fabricate, falsify, or selectively present
  measurements.
* **Fair comparisons.** Implement each programming-model variant (e.g. CUDA,
  HIP, SYCL, OpenMP, and any other supported programming system) in good faith so
  that cross-vendor and cross-model comparisons remain fair and meaningful. Avoid
  intentionally handicapping any vendor, device, language, or model.
* **Correctness.** Keep the reference/verification logic of a benchmark intact so
  that results can be validated.

## Contribution, Review, and Merge Process

To keep HeCBench a healthy and collaborative project, all contributors are
expected to follow the process below in addition to the standards above:

* **Pull Requests.** Each PR should have a clear description of the motivation
  and the changes made, and should reference any related issue if it exists.
* **Review.** A PR requires at least one approving review from a project
  maintainer before it can be merged. Reviewers and authors should
  communicate respectfully and constructively, in line with this Code of
  Conduct.
* **Merge.** Only project maintainers merge PRs, and only after required reviews
  and, where applicable, the project's continuous integration / continuous
  delivery (CI/CD) checks pass. As ORNL's CI/CD infrastructure is rolled out,
  contributors are expected to keep their changes building and passing the
  automated checks, and not to bypass, disable, or tamper with these checks.
* **Large files and DVC.** Large files should be managed through Data Version
  Control (DVC) rather than committed directly to the Git repository. External
  contributors are not expected to upload files directly to the project's DVC
  remote or S3 backend. Instead, the PR description should provide links from
  which maintainers can retrieve the files, along with enough information to
  identify their purpose, origin, licensing, and expected contents. During
  review, maintainers will inspect the files and, when appropriate, add them to
  the DVC-managed storage as part of the PR review and integration process.
* **New benchmarks.** When contributing a new benchmark, include the required
  attribution and licensing information for any upstream source, and provide the
  corresponding programming-model variants and verification where applicable.
* **Issues.** Use issues to report bugs, request features, or discuss design
  decisions before opening large PRs.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.1, available at
[https://www.contributor-covenant.org/version/2/1/code_of_conduct.html][v2.1].

Community Impact Guidelines were inspired by
[Mozilla's code of conduct enforcement ladder][mozilla].

For answers to common questions about this code of conduct, see the FAQ at
[https://www.contributor-covenant.org/faq][faq]. Translations are available at
[https://www.contributor-covenant.org/translations][translations].

[homepage]: https://www.contributor-covenant.org
[v2.1]: https://www.contributor-covenant.org/version/2/1/code_of_conduct.html
[mozilla]: https://github.com/mozilla/diversity
[faq]: https://www.contributor-covenant.org/faq
[translations]: https://www.contributor-covenant.org/translations
