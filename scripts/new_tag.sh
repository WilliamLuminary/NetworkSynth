#!/usr/bin/env bash
#
# Cut the next release tag.
#
#     scripts/new_tag.sh [--dev] [--major|--minor|--patch] [--push]
#     scripts/new_tag.sh --dev -0.1.0 --push        # same thing, level as a number
#
# The number is always one step past the highest v* tag, whether that tag was a
# release or a pre-release, so versions never collide or go backwards. --dev adds
# the -dev suffix, which is what sends the publish to dist-dev instead of dist.
# Without a level, the patch digit moves.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DEV=""
LEVEL="patch"
PUSH="no"

usage() {
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dev) DEV="-dev" ;;
        --major|--minor|--patch) LEVEL="${1#--}" ;;
        -1.0.0) LEVEL="major" ;;
        -0.1.0) LEVEL="minor" ;;
        -0.0.1) LEVEL="patch" ;;
        --push) PUSH="yes" ;;
        --create) ;;  # accepted and ignored: creating is what this script does
        -h|--help) usage 0 ;;
        *) echo "new_tag: unknown argument '$1'" >&2; usage 1 ;;
    esac
    shift
done

# The highest number any v* tag has reached, ignoring the -dev suffix.
latest="$(git tag -l 'v*' | sed -e 's/-dev$//' -e 's/^v//' \
          | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)"
latest="${latest:-0.0.0}"

IFS=. read -r major minor patch <<< "$latest"
case "$LEVEL" in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
esac

tag="v${major}.${minor}.${patch}${DEV}"
commit="$(git rev-parse --short HEAD)"
branch="$(git rev-parse --abbrev-ref HEAD)"

if git rev-parse -q --verify "refs/tags/$tag" > /dev/null; then
    echo "new_tag: $tag already exists" >&2
    exit 1
fi

# The publish workflow refuses a release tag that is not an ancestor of main, so
# fail here rather than after a push that cannot be undone quietly.
if [ -z "$DEV" ] && ! git merge-base --is-ancestor HEAD origin/main 2>/dev/null; then
    echo "new_tag: $commit is not on origin/main, so $tag would fail to publish." >&2
    echo "         Use --dev for a pre-release, which publishes from any branch." >&2
    exit 1
fi

git tag -a "$tag" -m "NetworkSynth $tag"
echo "$tag -> $commit ($branch), was $latest"

if [ "$PUSH" = "yes" ]; then
    git push origin "$tag"
    echo "pushed; the workflow publishes to $([ -n "$DEV" ] && echo dist-dev || echo dist)"
else
    echo "push with: git push origin $tag"
fi
