#!/usr/bin/env bash
# Gate das notas de release: o CHANGELOG.md precisa ter uma entrada datada para a versão
# da tag e docs/releases/<tag>.md precisa existir e não pode ser um rascunho. As notas são
# lidas do commit tagueado, então o que foi revisado é o que vai para a release.
#
# Uso: verify_release_notes.sh <tag> [<raiz-do-repositório>]
set -euo pipefail

usage="usage: verify_release_notes.sh <tag> [<repository-root>]"
tag="${1:?${usage}}"
root="${2:-.}"

version="${tag#v}"
changelog="${root}/CHANGELOG.md"
notes="${root}/docs/releases/${tag}.md"

if [[ ! -f "${changelog}" ]]; then
  echo "::error::CHANGELOG.md was not found." >&2
  exit 1
fi

pattern="^## \\[${version//./\\.}\\] - [0-9]{4}-[0-9]{2}-[0-9]{2}\$"
if ! grep -Eq "${pattern}" "${changelog}"; then
  echo "::error::CHANGELOG.md has no dated entry '## [${version}] - YYYY-MM-DD'." >&2
  exit 1
fi

if [[ ! -f "${notes}" ]]; then
  echo "::error::Release notes docs/releases/${tag}.md were not found." >&2
  exit 1
fi

if grep -Eq '^> \*\*Status: rascunho' "${notes}"; then
  echo "::error::docs/releases/${tag}.md is still a draft (Status: rascunho)." >&2
  exit 1
fi

echo "Changelog entry and release notes for ${tag} are in place."
