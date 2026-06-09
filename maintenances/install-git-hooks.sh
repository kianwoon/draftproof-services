#!/usr/bin/env bash
#
# Activate the local branch-cleanup git hooks by pointing core.hooksPath at the
# tracked maintenances/githooks directory. core.hooksPath is stored in the
# shared (common) git config, so it applies to every worktree of this repo.
#
# An ABSOLUTE path is used so worktrees checked out at older commits (which may
# not contain maintenances/) still resolve the hooks.
#
# Install:    maintenances/install-git-hooks.sh
# Uninstall:  git config --unset core.hooksPath
#
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hooks_dir="${repo_root}/maintenances/githooks"

if [ ! -d "$hooks_dir" ]; then
  echo "install-git-hooks: ${hooks_dir} not found. Run from a worktree that has maintenances/." >&2
  exit 1
fi

chmod +x "${hooks_dir}/post-merge" "${hooks_dir}/pre-push" "${repo_root}/maintenances/reap-merged-branches.sh"
git config core.hooksPath "$hooks_dir"

echo "Installed branch-cleanup hooks."
echo "  core.hooksPath = $(git config --get core.hooksPath)"
echo "  hooks: post-merge, pre-push"
echo "Uninstall with: git config --unset core.hooksPath"
