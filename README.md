# frc

Fourier Ring Correlation (FRC) and resolution determination for microscopy images, from two images or from one image of counts split in two (1FRC). This repository holds three packages, released together at one version:

| Package | Where | What it is |
|---|---|---|
| [`frc`](python/frc) | [PyPI](https://pypi.org/project/frc/) | The Python library: `one_frc`, `two_frc`, thresholds and resolution, on NumPy arrays. |
| [`frc-core`](python/frc-core) | [PyPI](https://pypi.org/project/frc-core/) | The compiled core, without dependencies: arrays in through the buffer protocol, `memoryview`s out. `frc` builds on it. It replaces rustfrc. |
| [`frc`](crates/frc) | [crates.io](https://crates.io/crates/frc) | The Rust library that both Python packages are built from. |

Most users want the first: `pip install frc`, and see [its README](python/frc/README.md).

## Layout

- `crates/frc`: the Rust library. `binomial.rs` holds the split, `frc.rs` the FRC, and `platform.rs` the crate's unsafe code.
- `crates/frc-python`: the PyO3 bindings, built into the module `frc_core._internal`, and never published to crates.io. `buffers.rs` reads arrays through the buffer protocol and makes the `memoryview`s, and `buffer_memory.rs` holds its only unsafe code.
- `python/frc-core`: the Python package frc-core, built with maturin from `crates/frc-python`. `frc_core/_internal.pyi` holds the module's types.
- `python/frc`: the Python package frc, pure Python, built with uv.
- `docs/PERFORMANCE.md`: where the time goes, what was tried, and the literature. `docs/history` keeps rustfrc's changelog.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Rust toolchain. Then:

- `uv sync --all-packages` creates `.venv` with both Python packages and their development dependencies, and builds the extension. Run it again after changing the Rust code. Pass `-p 3.11` to develop against the oldest supported Python.
- `uv run pytest` runs the tests of both Python packages.
- `cargo test --workspace` runs the Rust tests, and `cargo bench -p frc` times the split and the FRC at several sizes.

## Releasing

Releases go through [touchgate](https://github.com/borink-org/touchgate) and publish all three packages at one version. Every pull request adds a line under `## Unreleased` in `CHANGELOG.md`. `release.yml` publishes only from the `publish` branch, with the wheels that `wheels.yml` builds. A change to either file goes on `main`, and the repository owner copies it to `publish`.

## License

The crate frc and frc-core are licensed under Apache-2.0, and the Python library frc under MIT; each package directory holds its license.
