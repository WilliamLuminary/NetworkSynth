#!/usr/bin/env bash
#
#     scripts/publish_dist.sh [source-ref] [tag]
#
# Built from a commit through a temporary index, so the working tree is never read.
# Nothing is pushed.

set -euo pipefail

BRANCH="${DIST_BRANCH:-dist}"
SOURCE="${1:-HEAD}"
TAG="${2:-}"

# 'source:destination' publishes a path under a different name.
PATHS=(
    src
    pyproject.toml
    requirements.txt
    README_dist.md:README.md
    LICENSE
    .gitignore
)

cd "$(git rev-parse --show-toplevel)"
source_commit="$(git rev-parse --verify "${SOURCE}^{commit}")"

index="$(mktemp)"
trap 'rm -f "$index"' EXIT
export GIT_INDEX_FILE="$index"
git read-tree --empty

for spec in "${PATHS[@]}"; do
    path="${spec%%:*}"
    destination="${spec##*:}"
    entry="$(git ls-tree "$source_commit" -- "$path")"
    if [ -z "$entry" ]; then
        echo "publish_dist: $path is not in $SOURCE" >&2
        exit 1
    fi
    if [ "$(echo "$entry" | awk '{print $2}')" = "tree" ]; then
        git read-tree --prefix="$destination/" "$source_commit:$path"
    else
        git update-index --add --cacheinfo "$(echo "$entry" | awk '{print $1","$3}'),$destination"
    fi
done

# A consumer with no tags has no 'git describe' to ask, so the number is baked in.
if [ -n "$TAG" ]; then
    version="${TAG#dist-}"
    version="${version#v}"
    init="src/networksynth/__init__.py"
    stamped="$(git show "$source_commit:$init" \
               | sed -E "s/^__version__ = \".*\"\$/__version__ = \"$version\"/")"
    if ! printf '%s' "$stamped" | grep -q "^__version__ = \"$version\"\$"; then
        echo "publish_dist: no __version__ line to stamp in $init" >&2
        exit 1
    fi
    blob="$(printf '%s\n' "$stamped" | git hash-object -w --stdin)"
    git update-index --add --cacheinfo "100644,$blob,$init"
    echo "stamped $init with $version"
fi

tree="$(git write-tree)"
message="dist from $(git rev-parse --short "$source_commit"): $(git log -1 --format=%s "$source_commit")"

# A release whose code did not change still gets its tag, on the commit already there.
if parent="$(git rev-parse --verify --quiet "refs/heads/$BRANCH")"; then
    if [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
        echo "publish_dist: $BRANCH already carries this tree."
        commit="$parent"
    else
        commit="$(git commit-tree "$tree" -p "$parent" -m "$message")"
        git update-ref "refs/heads/$BRANCH" "$commit"
    fi
else
    commit="$(git commit-tree "$tree" -m "$message")"
    git update-ref "refs/heads/$BRANCH" "$commit"
fi

if [ -n "$TAG" ]; then
    git tag -a "$TAG" "$commit" -m "NetworkSynth $TAG"
fi

file_count="$(git ls-tree -r --name-only "$commit" | wc -l)"
byte_count="$(git ls-tree -r -l "$commit" | awk '{total += $4} END {print total}')"
echo "$BRANCH -> $(git rev-parse --short "$commit")  ($file_count files, $((byte_count / 1024)) KB)"
echo "push with: git push origin $BRANCH${TAG:+ $TAG}"
