# Contributing to frc

This repository holds the Python library frc, its compiled core frc-core, and the Rust crate frc. [README.md](README.md) describes what they compute.

## Building and testing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Rust toolchain. Then:

- `uv sync --all-packages` creates `.venv` with both Python packages and their development dependencies, and builds the extension. Run it again after you change the Rust code. Pass `-p 3.11` to develop against the oldest supported Python.
- `uv run pytest` runs the tests of both Python packages.
- `cargo test --workspace` runs the Rust tests, and `cargo bench -p frc` times the split and the FRC at several sizes.
- CI also runs `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, and `cargo doc --workspace --no-deps --locked` with `RUSTDOCFLAGS` set to `-D warnings`.

Every pull request adds a line under `## Unreleased` in `CHANGELOG.md`.

To build a wheel of frc-core, run `uvx maturin build --release` in `python/frc-core`. It builds one wheel for CPython 3.11 and later. For free-threaded Python, build with a free-threaded interpreter, for example `uvx maturin build --release -i python3.15t`. The wheels land in `target/wheels` at the root of the repository.

## Layout

- `crates/frc`: the Rust library. `binomial.rs` holds the split, `frc.rs` the FRC, `rings.rs` and `window.rs` the rings and the window, `cpu.rs` the choice of CPU instructions, and `platform.rs` the crate's unsafe code.
- `crates/frc-python`: the PyO3 bindings, built into the module `frc_core._internal`, and never published to crates.io. `buffers.rs` reads arrays through the buffer protocol and makes the `memoryview`s, and `buffer_memory.rs` holds its only unsafe code.
- `python/frc-core`: the Python package frc-core, built with maturin from `crates/frc-python`. `frc_core/_internal.pyi` holds the module's types.
- `python/frc`: the Python package frc, pure Python, built with uv.

`crates/frc` and `python/frc` link to the license files at the root, and `python/frc-core`, whose wheels are also built on Windows, keeps copies of them.

## Releasing

Releases go through [touchgate](https://github.com/borink-org/touchgate) and publish all three packages at one version: the crate to crates.io, and frc-core before frc to PyPI.

1. Run `touchgate prepare <version>` on `main` and push it, or start the `touchgate` workflow with the version and merge its pull request. It dates `## Unreleased` and sets the version everywhere, including frc's requirement on frc-core and both lock files.
2. On the same day or the next, run `touchgate approve`. It signs an approval of the tip of `main` with the security key and starts `release.yml`, which builds and tests every file with `wheels.yml`, waits for the commit's checks, publishes, and tags the release last. If it fails, re-run its failed jobs; publishing skips what a registry has already.

`release.yml` and `wheels.yml` run from the `publish` branch, and the jobs that publish run in the `release` environment, which admits only that branch. crates.io and PyPI trust only that workflow in that environment. A change to either file goes on `main` and is then copied to `publish`.
