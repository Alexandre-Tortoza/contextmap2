# Bundle portátil do `ContextMapArtifact`

Este documento descreve `src/contextmap/artifact/serialization/bundle.py` (issue #159). O layout do artifact está em [`storage-layout.md`](storage-layout.md) e a validação em [`integrity-validation.md`](integrity-validation.md).

Um `ContextMapArtifact` **referencia** os artifacts de que depende (a geometria, e evidência opcional) em vez de copiá-los. Movê-lo sozinho mantém tudo que não precisa de geometria legível, mas deixa a geometria para trás. Um **bundle** é um diretório que leva o artifact e, por uma política de fechamento explícita, os artifacts a montante de que ele precisa, para que possa ser movido para outro filesystem e aberto lá sem nenhum caminho do workspace original. Não há upload em nuvem nem formato de arquivo compactado: é um diretório (um `tar` dele é só transporte, como em `storage-layout.md`).

Exportar é **transporte, não inferência**: o bundle não finge ser um run científico novo.

```python
manifest = export_bundle(
    artifact_dir,
    output_dir,
    policy=ClosurePolicy.REQUIRED,
    evidence=[("semantic_fusion_run", "run-0003")],  # só com SELECTED_EVIDENCE
    dependency_paths={"corridor-02--run-0001": Path("…")},  # se o workspace foi movido
)
problems = verify_bundle(output_dir)  # () quando íntegro
```

## Políticas de fechamento

| Política | Leva | Deixa de fora (registrado, com o motivo) |
| --- | --- | --- |
| `CORE_ONLY` (`core-only`) | só o artifact | toda dependência, obrigatória ou não: "left out by the core-only policy". O bundle não resolve geometria sozinho. |
| `REQUIRED` (`core+required`) | o artifact e toda dependência **obrigatória** (a geometria) | evidência opcional: "optional evidence was not selected" |
| `SELECTED_EVIDENCE` (`core+selected-evidence`) | o fechamento obrigatório e a evidência opcional escolhida (`evidence=[(tipo, id)]`) | a evidência opcional não escolhida |

Escolher evidência com outra política, ou uma dependência que o artifact não registra, é `BundleError`. Nada é omitido em silêncio: o manifest do bundle lista `embedded` e `omitted`.

## O que bloqueia a exportação

O artifact de origem é **verificado por inteiro** (`ValidationLevel.FULL`) antes de qualquer cópia, com qualquer política. Um erro do validador (hash diferente, arquivo ausente ou truncado, índice quebrado, **dependência obrigatória ausente ou que não é a registrada**, linhagem incompatível…) bloqueia a exportação com um `BundleError` que lista os códigos, e nada é publicado. Evidência opcional escolhida que não pode ser achada também bloqueia: quem a escolheu quer levá-la. Um artifact que não pode ser verificado na origem não é exportado, nem como `core-only`.

## O que é copiado

- **O artifact**: todos os arquivos do inventário byte a byte, mais o `README.md`, em `artifact/`. A identidade de conteúdo, o inventário, `written_at`, `context_map_id` e todos os digests das dependências ficam **idênticos**. A única coisa reescrita é a dica `locator` de cada dependência (transporte, fora da identidade): `../dependencies/<tipo>/<id>` para a que foi levada, `null` para a que não foi (a dica antiga apontaria para o workspace de origem). O exportador confere que a identidade recalculada continua igual.
- **Cada dependência levada**, em `dependencies/<tipo>/<id>/`: o `manifest.json` e os arquivos do **seu** inventário. Nunca `debug/`, nunca arquivo fora do inventário (rosbag, checkpoint, sobra), então não há como levar dados de debug ou pesos por acidente.
- Cada arquivo é copiado em blocos, com SHA-256 calculado durante a leitura e **conferido contra o inventário de onde veio**; se a origem mudar durante a exportação, ela é recusada e nada é publicado.

A publicação é atômica (`AtomicRunDirectory`): um diretório de bundle nunca aparece pela metade, e um bundle existente nunca é sobrescrito (`ArtifactExistsError`).

## Layout e manifest do bundle

```text
<bundle>/
├── manifest.json            # artifact_type "context_map_bundle"; política, origem, embedded/omitted, inventário
├── README.md                # resumo determinístico
├── artifact/                # o ContextMapArtifact
└── dependencies/<tipo>/<id>/
```

O `manifest.json` do bundle usa `"artifact_type": "context_map_bundle"`, então **um bundle é distinguível do artifact que carrega**: `ContextMapArtifactReader.open(bundle)` recusa com "is a bundle, not an artifact: open its 'artifact' directory", e o artifact carregado (`artifact/`) abre como qualquer outro e valida como `verified` quando o fechamento inclui a geometria. O manifest registra a política, a identidade de conteúdo e as versões do artifact de origem, `embedded` (identidade, digest, exigência, local, número e tamanho dos arquivos), `omitted` (identidade, digest, exigência e motivo), a versão do validador que verificou a origem, o inventário com hash de cada arquivo e `bundle_identity`. Ela é o SHA-256 de tudo isso **exceto** `exported_at`, então as mesmas entradas dão o mesmo bundle (só o horário muda), e políticas diferentes dão identidades diferentes. O manifest não tem run, índice de run nem configuração: não é uma execução.

## Verificação

`verify_bundle(bundle)` devolve os problemas (vazio quando íntegro): o hash de cada arquivo contra o inventário do bundle, a identidade recalculada, que o artifact carregado ainda tem a identidade de conteúdo que o bundle registra, e a validação completa do artifact. Uma dependência que o bundle declara como omitida não é problema: é o que a política disse. Nada é reparado.

## Validação

`tests/artifact/test_context_map_serialization_bundle.py`: cada política, distinção entre bundle e artifact, abertura depois de relocar o bundle sem nenhum arquivo do workspace original (geometria resolvida de dentro do bundle), evidência escolhida, recusas (seleção inexistente, com outra política, ausente; dependência obrigatória ausente com cada política; origem danificada), identidades preservadas e só as dicas reescritas, origem e dependências intocadas, nada de `debug/` ou arquivo solto, conferência dos bytes copiados (uma origem que muda durante a exportação), sobrescrita recusada, determinismo, origem movida com `dependency_paths`, e detecção de arquivo alterado no bundle e de manifest adulterado.
