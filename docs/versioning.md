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

Ao publicar uma tag que corresponda a `v*.*.*`, o workflow de release valida o formato da versão, gera a distribuição Python, publica o build como artifact do workflow e cria uma GitHub Release.

Releases da série `v0.x.y` são marcadas como pre-release automaticamente.

O repositório não publica no PyPI durante a fase de validação do canonical pipeline.
