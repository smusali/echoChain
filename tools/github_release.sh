#!/bin/sh
set -eu
if [ -z "${REF_NAME:-}" ]; then
  echo "REF_NAME is required" >&2
  exit 2
fi
case "$REF_NAME" in
  *rc*)
    gh release create "$REF_NAME" --title "$REF_NAME" --notes-file release-notes.md --prerelease
    ;;
  *)
    gh release create "$REF_NAME" --title "$REF_NAME" --notes-file release-notes.md
    ;;
esac
