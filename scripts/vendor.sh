#!/usr/bin/env bash
# Copy solver files into a consuming repo and record what was copied.
#
#   scripts/vendor.sh <dest-dir> <file>...
#   e.g. scripts/vendor.sh ../../pkg-py/mxlpy-native/c/vendor c/radau5.c licences/HAIRER-LICENCE.txt
#
# Files are given relative to this repo and land flat in <dest-dir>. <dest-dir>/VENDORED
# records the commit and a sha256 per file, which the consumer's tests check, so a
# vendored file can't be edited in place. The commit must be clean and pushed.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
dest=${1:?usage: vendor.sh <dest-dir> <file>...}
shift
[ $# -gt 0 ] || { echo "no files given" >&2; exit 1; }

if [ -n "$(git -C "$root" status --porcelain)" ]; then
    echo "mxl-solvers has uncommitted changes; commit and push them first" >&2
    exit 1
fi
sha=$(git -C "$root" rev-parse HEAD)
if ! git -C "$root" branch -r --contains "$sha" | grep -q 'origin/main'; then
    echo "warning: $sha isn't on origin/main yet; push it before committing the vendored copy" >&2
fi

mkdir -p "$dest"
manifest="$dest/VENDORED"
{
    echo "# Vendored from https://github.com/Computational-Biology-Aachen/mxl-solvers at $sha."
    echo "# Don't edit these files here: change mxl-solvers, then re-run its scripts/vendor.sh."
} > "$manifest"
for f in "$@"; do
    name=$(basename "$f")
    cp "$root/$f" "$dest/$name"
    echo "$(sha256sum "$dest/$name" | cut -d' ' -f1)  $name" >> "$manifest"
done
echo "vendored $# files at ${sha:0:7} into $dest"
