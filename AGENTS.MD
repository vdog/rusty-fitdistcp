Always use the docker container for development tools.  Do not attempt to install them or search for them on the host system.

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
