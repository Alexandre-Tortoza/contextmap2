# Validação estrutural

Este documento descreve `src/contextmap/ingestion/validation.py`: checagens sobre o **conteúdo** de observações decodificadas, complementares às checagens de arquivo (`sequence_artifact.py`, issues #39/#46) e de calibração (`calibration.py`, issue #41).

## Convenção

Toda função aqui segue a mesma convenção já estabelecida por `SequenceArtifactReader.verify_integrity()`: retorna `list[str]` de problemas legíveis, nunca levanta exceção, lista vazia = nenhum problema. Nenhuma checagem aqui requer GPU, modelo, ou artefato persistido — operam diretamente sobre `Sequence[SourceObservation]` em memória, então rodam em qualquer PR sem dependências pesadas.

## Cobertura

| Função | O que detecta |
| --- | --- |
| `validate_image_observation()` | `width`/`height` não positivos; `len(data)` inconsistente com `width * height * bytes_por_pixel` do `encoding` declarado. |
| `validate_lidar_observation()` | `point_step_bytes` não positivo; `len(data)` inconsistente com `point_count * point_step_bytes`; nome de campo duplicado; `offset_bytes` de um campo fora do intervalo `[0, point_step_bytes)`. |
| `validate_timestamp_ordering()` | Timestamps não-monotônicos dentro do **mesmo** `clock_id` (observações de `clock_id` diferentes nunca são comparadas — mesma regra de `docs/synchronization.md`); duplicatas quando `allow_duplicates=False`. |
| `validate_frame_references()` | `frame_id` de uma observação que não aparece em nenhuma `CalibrationEntry.frame_id` nem em `CalibrationSet.static_transforms` (quando uma calibração é fornecida; sem calibração, não há nada para checar). |
| `validate_observations()` | Roda todas as anteriores sobre uma sequência completa. |

## Fixtures determinísticas

`tests/ingestion/fixtures.py` reúne o conjunto de fixtures desta issue: `build_valid_sequence()` (multimodal: RGB + point cloud + IMU + pose externa), `build_calibration_set()` (calibração/frame representativos, casando com os frames de `build_valid_sequence()`), e variantes intencionalmente inválidas para cada checagem acima (`build_image_with_data_size_mismatch()`, `build_lidar_with_inconsistent_fields()`, `build_non_monotonic_observations()`, `build_observation_with_unknown_frame()`). Tudo em Python puro, gerado em tempo de teste — nenhum arquivo binário versionado, mesma abordagem já usada pelos testes dos adapters ROS 1/ROS 2 (#44, #45).

Este módulo de fixtures é usado pelos novos testes desta issue (`test_validation.py`, `test_diagnostics.py`); os arquivos de teste já existentes de issues anteriores mantêm seus próprios builders locais — consolidá-los retroativamente não é escopo desta issue.
