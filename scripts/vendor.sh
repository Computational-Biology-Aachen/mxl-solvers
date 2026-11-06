#!/usr/bin/env bash
# Copy solver files into a consuming repo and record what was copied.
#
#   scripts/vendor.sh <dest-dir> <file-or-dir>...
#   e.g. scripts/vendor.sh ../../pkg-py/mxlpy-native/c/vendor c/radau5.c c/sundials licences/HAIRER-LICENCE.txt
#
# Files are given relative to this repo. They land in <dest-dir> at their path below c/ or
# licences/, so c/radau5.c becomes <dest-dir>/radau5.c and c/sundials/src/cvode/cvode.c
# becomes <dest-dir>/sundials/src/cvode/cvode.c. A directory stands for every file in it.
# <dest-dir>/VENDORED records the commit and a sha256 per file, which the consumer's tests
# check, so a vendored file can't be edited in place. VENDORED is rewritten on every run,
# so always pass the full list. The commit must be clean and pushed.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
dest=${1:?usage: vendor.sh <dest-dir> <file-or-dir>...}
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

files=()
for f in "$@"; do
    if [ -d "$root/$f" ]; then
        while IFS= read -r g; do files+=("${g#"$root"/}"); done < <(find "$root/$f" -type f | LC_ALL=C sort)
    else
        files+=("$f")
    fi
done

mkdir -p "$dest"
manifest="$dest/VENDORED"
{
    echo "# Vendored from https://github.com/Computational-Biology-Aachen/mxl-solvers at $sha."
    echo "# Don't edit these files here: change mxl-solvers, then re-run its scripts/vendor.sh."
} > "$manifest"
for f in "${files[@]}"; do
    name=${f#c/}
    name=${name#licences/}
    mkdir -p "$(dirname "$dest/$name")"
    cp "$root/$f" "$dest/$name"
    echo "$(sha256sum "$dest/$name" | cut -d' ' -f1)  $name" >> "$manifest"
done
echo "vendored ${#files[@]} files at ${sha:0:7} into $dest"
