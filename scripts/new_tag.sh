#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DEV=""
LEVEL="patch"
PUSH="no"
CREATE="no"

usage() {
    cat <<'TEXT'
Cut the next release tag.

    scripts/new_tag.sh --create [--dev] [--major|--minor|--patch] [--push]
    scripts/new_tag.sh --create --dev -0.1.0 --push     # level as a number

  --create   make the tag; without it this help is all that happens
  --dev      pre-release: adds -dev, publishes to dist-dev, any branch
  --push     push the tag, which is what starts the publish
  --major    -1.0.0
  --minor    -0.1.0
  --patch    -0.0.1, the default

The number is always one step past the highest v* tag, release or pre-release,
so the two share one sequence and can never collide. A release tag must be on
origin/main, or the publish workflow would refuse it.
TEXT
    exit "${1:-0}"
}

[ $# -eq 0 ] && usage 0

while [ $# -gt 0 ]; do
    case "$1" in
        --dev) DEV="-dev" ;;
        --major|--minor|--patch) LEVEL="${1#--}" ;;
        -1.0.0) LEVEL="major" ;;
        -0.1.0) LEVEL="minor" ;;
        -0.0.1) LEVEL="patch" ;;
        --push) PUSH="yes" ;;
        --create) CREATE="yes" ;;
        -h|--help) usage 0 ;;
        *) echo "new_tag: unknown argument '$1'" >&2; usage 1 ;;
    esac
    shift
done

[ "$CREATE" = "yes" ] || usage 0

# Highest v* number, release or pre-release, so the two share one sequence.
latest="$(git tag -l 'v*' | sed -e 's/-dev$//' -e 's/^v//' \
          | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' \
          | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"
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

# The workflow refuses a release tag off main, so fail before the push, not after.
if [ -z "$DEV" ] && ! git merge-base --is-ancestor HEAD origin/main 2>/dev/null; then
    echo "new_tag: $commit is not on origin/main, so $tag would fail to publish." >&2
    echo "         Use --dev for a pre-release, which publishes from any branch." >&2
    exit 1
fi

# Consumers get the stamped tree either way; this only keeps the drift visible.
declared="$(sed -nE 's/^__version__ = "(.*)"$/\1/p' src/networksynth/__init__.py)"
if [ "$declared" != "${tag#v}" ]; then
    echo "new_tag: __version__ is $declared, this tag is $tag." >&2
    echo "         The published tree gets ${tag#v} either way; main keeps $declared." >&2
fi

git tag -a "$tag" -m "NetworkSynth $tag"
echo "$tag -> $commit ($branch), was $latest"

if [ "$PUSH" = "yes" ]; then
    git push origin "$tag"
    echo "pushed; the workflow publishes to $([ -n "$DEV" ] && echo dist-dev || echo dist)"
else
    echo "push with: git push origin $tag"
fi
