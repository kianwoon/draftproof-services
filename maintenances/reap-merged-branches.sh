#!/usr/bin/env bash
#
# Delete local branches that are fully merged into the default branch.
# Local mirror of the GitHub branch-cleanup workflows.
#
# Deletes a branch only when ALL hold:
#   - it is not the default branch (main),
#   - it is not checked out in any worktree (git branch -d would refuse anyway),
#   - it is not sitting at the default branch's current tip (fresh / not stale),
#   - it is fully merged into the default branch (git branch --merged).
# Uses `git branch -d` (safe): git refuses if a branch is not fully merged.
#
# Usage:
#   maintenances/reap-merged-branches.sh            # delete merged branches
#   maintenances/reap-merged-branches.sh --dry-run  # show what would be deleted
#
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Resolve the default branch from origin/HEAD, falling back to main.
BASE="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')"
BASE="${BASE:-main}"

if ! git show-ref --verify --quiet "refs/heads/${BASE}"; then
  echo "reap: base branch '${BASE}' not found locally; nothing to do." >&2
  exit 0
fi

base_sha="$(git rev-parse "refs/heads/${BASE}")"

# Branches currently checked out in any worktree are protected.
worktree_branches="$(git worktree list --porcelain \
  | awk '/^branch /{ sub("refs/heads/","",$2); print $2 }')"

is_worktree_branch() {
  printf '%s\n' "$worktree_branches" | grep -qx -- "$1"
}

deleted=0
kept=0
while IFS= read -r br; do
  [ -z "$br" ] && continue
  [ "$br" = "$BASE" ] && continue
  if is_worktree_branch "$br"; then
    echo "  keep (worktree):  $br"; kept=$((kept + 1)); continue
  fi
  if [ "$(git rev-parse "refs/heads/$br")" = "$base_sha" ]; then
    echo "  keep (at tip):    $br"; kept=$((kept + 1)); continue
  fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "  would delete:     $br"
  else
    if git branch -d "$br" >/dev/null 2>&1; then
      echo "  deleted (merged): $br"; deleted=$((deleted + 1))
    else
      echo "  skip (-d refused): $br"; kept=$((kept + 1))
    fi
  fi
done < <(git branch --merged "refs/heads/${BASE}" --format='%(refname:short)')

echo "reap: base=${BASE} deleted=${deleted} kept=${kept} dry_run=${DRY_RUN}"
