# Releasing

One release publishes three packages at one version: the crate `frc` to crates.io, and the Python packages `frc-core` and `frc` to PyPI. `frc` on PyPI requires `frc-core` of exactly its own version. Releases go through [touchgate](https://github.com/borink-org/touchgate), which publishes only after an approval with the security key.

## Making a release

1. Start the `touchgate` workflow with the version, such as `0.2.0`, and merge the pull request it opens. It dates `## Unreleased` in `CHANGELOG.md`, and sets the version in `Cargo.toml`, in both `pyproject.toml` files, in frc's requirement on frc-core, and in `Cargo.lock` and `uv.lock`. Running `touchgate prepare 0.2.0` in a checkout does the same.
2. On the same day or the next, run `touchgate approve` in a checkout. It signs an approval of the tip of `main` with the security key and starts `release.yml` with it.

`release.yml` then checks the approval, checks the version and packages the crate, builds and tests every wheel and sdist with `wheels.yml`, and waits for every check on the commit. Two jobs publish: one to crates.io with `touchgate publish --to crates.io`, and one to PyPI with `touchgate publish --to pypi`, which uploads frc-core before frc. The tag and the GitHub release come last, so a tag always names a published release. If a release fails, re-run its failed jobs; publishing skips what a registry has already.

Every pull request that changes a package adds a line under `## Unreleased`.

## How the setup protects a release

- `release.yml` and `wheels.yml`, which it calls, run from the `publish` branch. Only the admin account, `tmtenbrink`, can change that branch. `main` keeps identical copies, because GitHub starts a workflow by hand only if the default branch has the file. A change to either file goes on `main`, and the admin copies it to `publish`.
- The jobs that publish run in the `release` environment, which admits only the `publish` branch.
- crates.io and PyPI trust only `release.yml` in that environment, through Trusted Publishing. No API token for any of the three packages exists.
- The jobs that build have no `id-token: write`, so they cannot get a token. The jobs that publish build nothing.

## Settings for the admin account

These are made once, with `tmtenbrink` in the browser. The first two exist from frc's earlier releases.

- A ruleset that lets only the admin create, update or delete `publish`, and blocks force pushes.
- A `release` environment that admits only the `publish` branch.
- On PyPI, the trusted publisher of `frc` for `release.yml` in the `release` environment, which exists, and a pending publisher of `frc-core` with the same workflow and environment.
- On crates.io, a trusted publisher of `frc` for `release.yml` in the `release` environment, with Trusted Publishing only mode on. crates.io takes a trusted publisher only for a crate that exists, so the crate's first version must be published by its owner before this.
- Copy `release.yml` and `wheels.yml` from `main` to `publish` once this layout is on `main`.
- Archive `tiptenbrink/rustfrc`, whose code moved here, after pointing its README here.

## Before the first release

- crates.io needs `frc` to exist before it trusts `release.yml`. Deciding how it gets there, such as a first publish by hand, is the owner's.
