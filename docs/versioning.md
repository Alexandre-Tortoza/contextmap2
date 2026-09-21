# Versionamento e releases

O ContextMap2 usa Semantic Versioning com tags no formato:

```text
vMAJOR.MINOR.PATCH
```

## Fase de validação

Enquanto o canonical pipeline estiver em validação, releases permanecem na versão major zero:

```text
v0.1.0
v0.2.0
v0.2.1
```

Use:

- `PATCH` para correções compatíveis, documentação e pequenas melhorias internas que não alteram intencionalmente o contrato do artefato;
- `MINOR` para novas capabilities, mudanças mensuráveis de pipeline ou evolução compatível do artefato/schema durante a fase de validação;
- `MAJOR` somente depois que existir um contrato externo estável e mudanças incompatíveis precisarem ser comunicadas formalmente.

## Origem das releases

Somente commits presentes em `main` podem receber tags de release.

O fluxo esperado é:

```text
issue branch
    ↓
milestone/*
    ↓
dev
    ↓
main
    ↓
vMAJOR.MINOR.PATCH
```

Uma branch de issue ou milestone nunca deve ser tagueada diretamente.

## Requisitos de release

Uma tag de release deve apontar para um commit que:

- passe a CI;
- tenha a evidência de avaliação relevante registrada;
- não possua corrupção conhecida de artefato ou regressão de serialização;
- documente mudanças incompatíveis de schema;
- seja reproduzível a partir da configuração e do código versionados;
- tenha sido promovido pelo fluxo normal até `main`.

## Automação

A tag é criada por um mantenedor; nenhum workflow cria tags. Ao receber uma tag que corresponda a `v*.*.*`, o workflow `Release` executa, em ordem, e só cria a GitHub Release se todas as etapas passarem:

1. `verify`: a tag tem o formato estrito `vMAJOR.MINOR.PATCH` e o commit tagueado é ancestral de `origin/main` (`.github/scripts/verify_release_tag.sh`, coberto por `tests/packaging/test_release_gate.py`). Se `main` não existir, o gate falha;
2. `checks`: reutiliza o workflow `CI` sobre o commit da tag (qualidade em Python 3.11, testes em 3.12 a 3.14, build de sdist e wheel com `twine check --strict`, smoke de instalação em ambientes novos e suíte completa contra a wheel instalada só com NumPy);
3. `publish`: instala a wheel que a CI construiu e testou, exige que ela reporte a versão da tag, gera `SHA256SUMS` e cria a release com `gh release create --verify-tag`, com as wheels, o sdist e os checksums. Somente este job tem `contents: write`; o workflow não usa segredos além do `GITHUB_TOKEN`.

Releases da série `v0.x.y` são marcadas como pre-release automaticamente. O passo de criação da release não tem execução a seco: ele só roda com uma tag real, e o restante do caminho (build, smoke, gate de tag) é exercitado a cada pull request pela CI.

O repositório não publica no PyPI durante a fase de validação do canonical pipeline.
