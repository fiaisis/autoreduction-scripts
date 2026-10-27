# AGENTS.md

Guidance for coding agents working in this repository.

## What this repo is

`autoreduction-scripts` is the repository of per-instrument Python reduction scripts (`<INSTRUMENT>/reduce.py`
and `<INSTRUMENT>/live_data.py`) used by the Flexible Interactive Automation (FIA) system. There is
no application package, `pyproject.toml`, or test suite here — only the instrument script
directories and a daily workflow (`.github/workflows/mari_sync.yml`) that checks upstream
`mantidproject/scriptrepository` for changes to `MARI/original.py`.

## Position in the FIA service

### Overarching FIA architecture

```text
[ISIS Instrument Archive (/archive, /imat) & lastrun.txt]
        │
        ▼
  file-watcher (Kopf Operator + Per-Instrument Watcher Pods)
        │  publishes file path to RabbitMQ: `watched-files`
        ▼
  run-detection ◄──── fetches instrument rules (`GET /instrument/{inst}/specification`) ────► FIA-API
        │  ingests .nxs metadata, verifies rules, publishes JobRequest JSON
        │  to RabbitMQ: `scheduled-jobs` (or retries via `failed-watched-files`)
        ▼
  jobcontroller (job_creator) ◄──── requests PreScript & posts Job ──► FIA-API ◄──► autoreduction-scripts (THIS REPO)
        │  spawns Kubernetes Job in cluster:
        ├──► Main container: runners (`ghcr.io/fiaisis/mantid` or `mantid-imaging`) -> writes reduced data to `/ceph`
        └──► Sidecar container: jobcontroller (job_watcher) -> updates job status/outputs in FIA-API (`PATCH /job/{id}`)

[Interactive / Web & Live Data Paths]
  scigateway (Host App) ──► frontend (FIA React Plugin)
                                ├──► fia-auth (UOWS login, JWT verify/refresh, `/experiments` RB lookup)
                                ├──► FIA-API (runs, jobs, scripts, specifications, reruns -> `watched-files` / `scheduled-jobs`)
                                └──► plotting-service (h5grove NeXus/HDF5 & IMAT image serving from `/ceph` & `/imat`)

  Kafka (`<INST>_events`, `<INST>_sampleEnv`, `<INST>_runInfo`)
        └──► live-data (Live Data Operator + Mantid Live Data Processor fetching `/live-data/{inst}/script` from FIA-API)

  LLSP (Long Lived Script Processors: `LLSP-API` + RabbitMQ + Celery `LLSP-Worker` built on `ghcr.io/fiaisis/mantid`)

[Infrastructure & Deployment]
  k8s (Terraform + Ansible for OpenStack clusters, Redpanda, local Minikube)
  gitops (ArgoCD App-of-Apps & Kustomize `base`/`envs`/`variants` manifests for all FIA services)
```

### This repository's location (upstream, downstream & siblings)

- **Downstream consumers**:
  - `FIA-API` (`../FIA-API`) fetches `<INSTRUMENT>/reduce.py` from this repository, applies the
    matching instrument `Transform` in `fia_api/scripts/transforms/` to inject run-specific
    parameters, and serves the resulting script to `jobcontroller` (`../jobcontroller`), which runs
    it inside a `runners` (`../runners`) container.
  - `FIA-API` also serves `<INSTRUMENT>/live_data.py` via `GET /live-data/{instrument}/script` to
    `live-data` (`../live-data`).
- **Critical cross-repo contracts**:
  - **Transform variable lines**: `FIA-API`'s transforms (`../FIA-API/fia_api/scripts/transforms/`)
    rewrite specific variable assignment lines in `<INSTRUMENT>/reduce.py`. Before renaming or
    reordering top-of-script parameter lines, check the corresponding transform in `../FIA-API`.
  - **`jobwatcher` stdout contract**: `reduce.py` scripts must print a single-line JSON object as
    their final stdout line (`{"status": "Successful", "status_message": ..., "output_files": [...], "stacktrace": ...}`)
    so `../jobcontroller`'s `jobwatcher` can parse the job outcome.
  - **`live-data` contract**: `live_data.py` scripts must define an `execute()` function called
    periodically by `../live-data`'s processor.

## Repository structure

```
<INSTRUMENT>/                One uppercase directory per instrument (e.g. MARI/, GEM/, IMAT/,
                              OSIRIS/, VESUVIO/, MERLIN/, ...), containing `reduce.py` and/or
                              `live_data.py`
.github/workflows/           Daily cron (`mari_sync.yml`) comparing upstream Mantid MARI template
                              against `MARI/original.py`
```

## Requirements for code changes

This repo has no `pyproject.toml`, linter config, or test suite — there is no automated unit/e2e
test or coverage infrastructure to run. Only perform script validation when `.py` scripts are
actually changed (skip validation for `.md`-only documentation changes). Before considering a `.py`
script change complete:
1. Manually validate the script's syntax and logic by running it (or the relevant portion of it)
   against the target `ghcr.io/fiaisis/mantid` or `ghcr.io/fiaisis/mantid-imaging` container
   (`../runners`), since that's the actual runtime environment these scripts execute in.
2. Check the change against the matching instrument `Transform` in
   `../FIA-API/fia_api/scripts/transforms/` to confirm the variable lines it rewrites still exist
   and are unaffected (see "Critical cross-repo contracts" above).
3. For `reduce.py` changes, confirm the final-line JSON stdout contract is preserved; for
   `live_data.py` changes, confirm `execute()` still exists with the expected signature.
4. Ask the user if you are unsure whether a given instrument's reduction behavior actually changed
   as intended — there's no test suite to catch a regression for you here.

## Issues, branches, commits & pull requests

- **Issues**: Use the org-wide default template from `fiaisis/.github`:
  ```shell
  gh issue create --template Issue
  ```
- **Branch naming**: `snake_case` description of the work without prefixes (`feature/`, `fix/`), e.g.
  `change_gem_path`.
- **Commit messages**: Write clear, imperative-mood commit summaries (e.g. `Use ENGINX new script
  for mantid 6.16`, `Adjust Vesuvio script to handle multiple files`).
- **Pull requests**:
  1. Validate syntax/execution against the target `ghcr.io/fiaisis/mantid` or
     `ghcr.io/fiaisis/mantid-imaging` container (and verify compatibility with the instrument's
     transform in `../FIA-API`).
  2. Use a clear sentence-case PR title and the org-wide default PR template from `fiaisis/.github`
     (`Closes # <Issue Number>` + `## Description`; ask the user if no issue number is evident).
  3. Attach an AI-assistance label to the PR (`gh label list` to check existing labels, or create
     `AI` via `gh label create "AI" --description "Created with the assistance of AI"`).

## General working practices & keeping this file up to date

- Prefer the `read` tool over `sed`/`cat`, and prefer the `write` and `edit` tools over `bash` or
  ad-hoc Python scripts when creating or modifying files.
- Ask the user when instrument parameters, mount paths (`/archive`, `/output`, `/extras`, `/imat`,
  `/gem`), or expected output filenames are ambiguous.
- If a change alters the structure or contracts described in this `AGENTS.md`, update this file as
  part of the same change.
