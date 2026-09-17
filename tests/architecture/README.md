# Architecture checks

Os testes deste diretório transformam boundaries documentados em validações rápidas de CI.

Eles analisam imports Python com `ast` e não importam módulos reais, portanto não carregam ROS, Torch ou modelos durante a validação.

## Regras atuais

O checker cobre:

- direção permitida entre capabilities;
- imports cross-module somente pela raiz pública `contextmap.<capability>`;
- proibição de acesso cross-module a `backends`, `infrastructure` e internals privados;
- proibição de capabilities de domínio dependerem de `runtime`;
- exceção explícita para a composition root importar backends concretos;
- SDKs pesados somente em diretórios designados de backend/infrastructure/adapter;
- scan automático de todos os arquivos Python existentes em `src/contextmap`.

## Atualizando uma boundary intencionalmente

Não alterar o allowlist apenas para fazer CI passar.

Quando uma mudança arquitetural legítima exige uma nova dependência:

1. atualizar a issue/decisão arquitetural que justifica ownership e direção;
2. atualizar `docs/architecture.md`, `docs/module-api.md` ou o documento específico afetado;
3. alterar `ALLOWED_DEPENDENCIES` neste teste no mesmo PR;
4. adicionar/ajustar fixture que demonstre o novo caso permitido e preserve violações próximas;
5. verificar que a mudança não cria ciclo no nível das capabilities;
6. registrar no PR por que o import público é necessário.

Se a dependência nova existe apenas porque um consumer precisa acessar internals de outro módulo, a correção esperada é normalmente ajustar o contrato/port público, não ampliar o allowlist.

## Adicionando uma capability

Ao criar uma capability real:

- adicionar seu nome em `CAPABILITIES`;
- declarar imports upstream permitidos em `ALLOWED_DEPENDENCIES`;
- manter a raiz pública conforme `docs/module-api.md`;
- adicionar pelo menos uma fixture se houver uma regra nova ou exceção específica.

## Backends e SDKs

`HEAVY_SDK_ROOTS` é uma proteção deliberadamente pequena. Um novo SDK pesado deve ser adicionado quando passar a existir no código e quando seu isolamento puder ser expresso de forma confiável.

Não transformar a lista em catálogo preventivo de bibliotecas hipotéticas.

## Escopo

Esses checks protegem imports e boundaries mecanicamente verificáveis. Eles não substituem code review para ownership semântico, schema, lineage ou comportamento científico.
