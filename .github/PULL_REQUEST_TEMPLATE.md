## Issue e milestone

- Issue: #
- Milestone:
- Branch de destino: `milestone/<slug>`

## Resumo

Descreva a mudança e por que ela é necessária.

## Tipo

- [ ] Feature
- [ ] Fix
- [ ] Research
- [ ] Experiment
- [ ] Refactor
- [ ] Testes / benchmark
- [ ] Documentação
- [ ] CI / manutenção

## Decisão de implementação

Explique a abordagem adotada, os principais trade-offs e qualquer decisão arquitetural que precise permanecer compreensível no futuro.

## Validação

Descreva como a alteração foi validada. Inclua testes, métricas, benchmarks, fixtures ou artefatos gerados quando a mudança afetar a qualidade do mapa.

## Impacto em contratos e artefatos

- [ ] Sem alteração de contrato/schema
- [ ] Alteração compatível de contrato/schema
- [ ] Alteração incompatível de contrato/schema

Se houver alteração, descreva os campos afetados, provenance, compatibilidade, migração e validação realizada.

## Impacto arquitetural e documental

- [ ] Sem alteração arquitetural pública
- [ ] Arquitetura/documentação atualizada no mesmo PR
- [ ] Documentação específica do módulo atualizada
- [ ] `docs/README.md` atualizado quando um novo ponto de entrada documental foi criado

Descreva decisões relevantes quando aplicável.

## Checklist

- [ ] A branch segue `<type>/<issue-number>-<slug>`
- [ ] Este PR aponta para a branch da milestone, não diretamente para `dev` ou `main`
- [ ] Commits seguem Conventional Commits e possuem corpo descritivo
- [ ] `make check` passa localmente
- [ ] Testes cobrem o comportamento relevante
- [ ] Docstrings públicas estão em inglês e seguem Google style
- [ ] Comentários explicativos e documentação Markdown estão em PT-BR
- [ ] Clean Code, SOLID, KISS, DRY e YAGNI foram aplicados sem abstração especulativa
- [ ] Comportamento experimental está isolado do caminho validado
- [ ] Nenhuma suposição específica de dataset foi introduzida sem documentação
