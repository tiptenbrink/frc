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

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Rust toolchain. Then:

- `uv sync --all-packages` creates `.venv` with both Python packages and their development dependencies, and builds the extension. Run it again after changing the Rust code. Pass `-p 3.11` to develop against the oldest supported Python.
- `uv run pytest` runs the tests of both Python packages.
- `cargo test --workspace` runs the Rust tests, and `cargo bench -p frc` times the split and the FRC at several sizes.

## Releasing

Releases go through [touchgate](https://github.com/borink-org/touchgate) and publish all three packages at one version: the crate to crates.io, and frc-core before frc to PyPI.

1. Run `touchgate prepare <version>` on `main` and push it, or start the `touchgate` workflow with the version and merge its pull request. It dates `## Unreleased` and sets the version everywhere, including frc's requirement on frc-core and both lock files.
2. On the same day or the next, run `touchgate approve`. It signs an approval of the tip of `main` with the security key and starts `release.yml`, which builds and tests every file with `wheels.yml`, waits for the commit's checks, publishes, and tags the release last. If it fails, re-run its failed jobs; publishing skips what a registry has already.

`release.yml` and `wheels.yml` run from the `publish` branch, and the jobs that publish run in the `release` environment, which admits only that branch. crates.io and PyPI trust only that workflow in that environment. A change to either file goes on `main` and is then copied to `publish`. Every pull request adds a line under `## Unreleased` in `CHANGELOG.md`.

## License

All three packages are licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. `crates/frc` and `python/frc` link to both files, and `python/frc-core`, whose wheels are also built on Windows, keeps copies of them.
