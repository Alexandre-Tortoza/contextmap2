#!/usr/bin/env bash
# Gate de release: a tag precisa ter o formato vMAJOR.MINOR.PATCH e apontar para um
# commit que já está em main (docs/versioning.md). O gate só lê o repositório; nunca
# cria tag, release nem branch.
#
# Uso: verify_release_tag.sh <tag> <commit> [<ref-de-main>]
set -euo pipefail

usage="usage: verify_release_tag.sh <tag> <commit> [<main-ref>]"
tag="${1:?${usage}}"
commit="${2:?${usage}}"
main_ref="${3:-refs/remotes/origin/main}"

if [[ ! "${tag}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "::error::Invalid release tag '${tag}': expected vMAJOR.MINOR.PATCH." >&2
  exit 1
fi

if ! git rev-parse --verify --quiet "${main_ref}^{commit}" > /dev/null; then
  echo "::error::Branch main was not found (${main_ref}); releases are only cut from main." >&2
  exit 1
fi

if ! git merge-base --is-ancestor "${commit}" "${main_ref}"; then
  echo "::error::Commit ${commit} is not on main; promote it through dev -> main before tagging ${tag}." >&2
  exit 1
fi

echo "Release tag ${tag} points to ${commit}, which is on main."
