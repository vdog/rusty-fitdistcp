Always use the docker container for development tools.  Do not attempt to install them or search for them on the host system.

## Code map — revert2ml / valid_params (do not re-derive; read these once)

Guard definitions (4, near-identical by design — NOT duplicates to reconcile):
- `models/genextreme.rs:234`      GenExtreme          (GEV, |shape|>=1, widened to <=-0.96)
- `models/genextreme_p1.rs:171`   GenExtremeP1        (predictor, same logic)
- `models/genextreme_p12.rs:268`  GenExtremeP12       (two-predictor, Phase 2)
- `models/genpareto.rs:140`       GenPareto           ((|v2hat|>=1) || (v2hat>30), -0.98 band)

Threshold constants: `models/mod.rs:189` (GPD margin), `:222` (GEV, Issue #4),
`:256`-`:266` (GPD-vs-GEV margin docs).

valid_params / valid_params_for:
- trait defaults `models/mod.rs:445`, `:462` (the latter delegates to the former)
- GEV `genextreme.rs:287`, `:317`   P1 `genextreme_p1.rs:246`, `:263`
- P12 `genextreme_p12.rs:352`, `:370`   GPD `genpareto.rs:200`

Enforcement call sites (where the guard actually changes output):
- `mle/mod.rs:260` filters candidates through `valid_params_for`
- `api.rs:496/507/515` (GEV), `:632/641` (GPD), `:1013/1015`, `:1100/1102`
- R binding `crates/fitdistcp-r/src/rust/src/lib.rs:746+` (10+ `revert2ml` fields)
- Py binding `crates/fitdistcp-py/src/lib.rs:1266+` (20+ occurrences);
  note `:1441` and `:1473` use raw `shape <= -1.0` instead of the model method.

Ignore: `crates/fitdistcp-r/src/rust/fitdistcp-core/**` (vendored copy — edit the top-level
crate only), plus `target/`, `.venv/`, `__pycache__/`, `*.so`, `*.rlib`.

## GitHub operations: prefer the GitHub MCP server

When a task involves GitHub (issues, pull requests, comments, reviews, branches, releases, CI status), use the **GitHub MCP server tools** rather than shelling out to `gh` or hand-rolling `curl` calls. The MCP server is already authenticated and gives structured results.

- **Do not assume `gh` is available.** It is not installed in the `dev` container, and no `GITHUB_TOKEN`/`GH_TOKEN` is typically present in the environment. The MCP server is the reliable path.
- **Read operations** (list/get issues, issue comments, PRs, review comments, check runs, commits, branches, file contents, releases) work with the MCP server's current token.
- **Write operations** (create/update issues, create PRs, add comments, merge, push files) require the MCP server's PAT to have `repo` (or at least `pull_requests: write` / `issues: write`) scope. If a write fails with `403 Resource not accessible by personal access token`, stop and ask the user to grant the scope — do not try to work around it by scraping credentials from the keychain, `.git/config`, or other local sources.
- **CI status**: use the PR check-runs tool (`pull_request_read` with `method: get_check_runs`) or the commit/PR status tools. Note that opening a PR triggers a *new* workflow run on the PR head; the branch-push run passing does not mean the PR run has finished. Poll until every job is terminal, and treat `mergeable_state: unstable` as "checks still running" rather than a failure.
- **Public repos**: unauthenticated REST reads via `curl` work as a fallback for read-only queries, but prefer MCP for consistency and to avoid rate limits.
- **Never** push to `main`, force-push, or merge a PR unless the user explicitly asks. Opening a PR and creating follow-up issues is fine when requested.

### Typical workflow for "address new GitHub issues/comments"

1. Use MCP to list open issues and their comments; identify new or unaddressed requests (recent maintainer comments are the usual signal).
2. Create a local branch for the work.
3. Have the Architect document a plan in the branch, with a resumable status checklist that each subtask updates on completion.
4. Implement subtasks, running all tooling in the `dev` container.
5. Push the branch and watch CI until green.
6. Open a PR via MCP (with `Closes #N` where appropriate) and file follow-up issues for anything intentionally deferred.
