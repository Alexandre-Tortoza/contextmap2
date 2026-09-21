# Instalação

Este documento descreve como instalar o ContextMap2, o que cada extra opcional traz e o que **não** vem de nenhum extra. O estado descrito é o de `origin/dev` em 2026-09-21; capabilities e backends que outras milestones ainda vão integrar (runtime real, entity resolution, `artifact`) não aparecem aqui como instaláveis.

## Requisitos

- Python 3.11 ou superior (`requires-python = ">=3.11"`). As versões declaradas nos classifiers são as executadas pela CI.
- NumPy abaixo de 2.4 (`numpy<2.4`): a partir da 2.4 os stubs usam alias `type` do PEP 695, que o mypy não analisa com `python_version = "3.11"`. É a única dependência da instalação base.

O pacote **não é publicado no PyPI** durante a fase de validação ([versioning.md](versioning.md)). Instale a partir de um clone ou da wheel anexada a uma GitHub Release (confira o `SHA256SUMS` da release):

```bash
python -m pip install .
python -m pip install contextmap-X.Y.Z-py3-none-any.whl
```

## Instalação base

```bash
python -m pip install .
```

Instala só o NumPy. É suficiente para importar todos os contratos públicos das capabilities e para abrir e validar artifacts persistidos (`SequenceArtifact`, `PerceptionRunArtifact`, `GeometricMapArtifact`, `SensorAssociationRunArtifact`, `SemanticFusionRunArtifact`, ...) sem ROS, Torch, CUDA ou modelos. Nenhum módulo importa um SDK opcional na importação, exceto os dois adapters de bag ROS (`ros1_bag` e `ros2_bag`), que ficam fora da API pública e só são alcançados pelo caminho pontilhado completo.

Duas verificações protegem essa promessa: `tests/packaging/test_dependencies.py` (todo import de terceiros está ligado a um extra ou listado como externo, e todos os módulos importam com qualquer módulo de terceiros, exceto NumPy, bloqueado) e `tests/packaging/smoke_installed_package.py` (executado em um venv novo com a wheel instalada).

## Extras opcionais

| Extra | Instala | Habilita | Verificação |
| --- | --- | --- | --- |
| `ros1` | `rosbags>=0.9,<1` | `Ros1BagSourceAdapter`, `Ros2BagSourceAdapter` (leem os dois formatos sem instalar o ROS) e o bag de entrada do wrapper do FAST-LIO | venv novo com a wheel e o extra, mais os 52 testes que dependem do `rosbags` |
| `ros2` | `rosbags>=0.9,<1` | o mesmo que `ros1`; existe para nomear a fonte | idem, com `ros2` |
| `vision` | `torch>=2.5`, `torchvision`, `transformers>=5.17,<6`, `Pillow>=10.0.1` | adapters DINOv2, DINOv3 e CLIP; autocast do SAM 3 | resolução de dependências (`pip install --dry-run`) em 2026-09-21; inferência real de DINOv2 e CLIP em 2026-09-20 com torch 2.14, transformers 5.17 e Pillow 12.3 |
| `dev` | ferramentas de qualidade (`ruff`, `mypy`, `pytest`, `pytest-cov`, `build`, `pre-commit`) e `rosbags` | `make check` | `make check` |

Notas sobre `vision`:

- `transformers` 5.x exige `torchvision` para o image processor; por isso ele faz parte do extra. Os pisos de `torch` e `Pillow` são os que o próprio `transformers` 5.x declara.
- O extra puxa o build padrão do PyTorch no PyPI. Para CPU puro ou para um build CUDA específico, instale o `torch` antes, do índice da PyTorch, e depois o extra.
- DINOv3 e SAM 3 usam repositórios *gated* no Hugging Face; o acesso é do usuário.

## O que não vem de nenhum extra

Estes runtimes são importados ou injetados pelo código, mas o PyPI não os distribui (consulta ao PyPI em 2026-09-21) ou o código só define a fronteira. Instalá-los é responsabilidade do usuário.

| Backend | Dependência | Como obter |
| --- | --- | --- |
| SAM 2 (`Sam2AutomaticMaskRuntime.from_model`) | módulo `sam2` | instalação a partir do repositório `facebookresearch/sam2` (não há `sam2`/`SAM-2` no PyPI) |
| AlphaCLIP | módulo `alpha_clip` e checkpoint | repositório `SunzeY/AlphaCLIP`; a origem e a licença dos checkpoints não foram verificadas |
| SAM 3, Qwen, Florence-2, Gemini, PTv3 | runtime ou cliente **injetado** por quem constrói o adapter | o código empacotado não importa o SDK; o runtime de PTv3 exige Pointcept, spconv e torch-scatter, que também não estão no PyPI como um pacote único |
| FAST-LIO | binário externo (ROS 1, normalmente em container) | fora do Python; o wrapper só escreve o bag de entrada e lê a trajetória |

## Verificando uma instalação

```bash
python -c "import contextmap; print(contextmap.__version__)"
python tests/packaging/smoke_installed_package.py          # base
python tests/packaging/smoke_installed_package.py --extra ros1
make smoke-install                                          # constrói, instala em venv novo e roda o smoke
```

O smoke exige que o pacote venha de uma instalação (não da árvore de fontes), que `__version__` seja a versão dos metadados e não o fallback `0.0.0`, que a instalação base dependa só do NumPy, que nenhum módulo pesado esteja presente sem o extra correspondente e que todos os módulos importem.

## Versão

A versão vem exclusivamente da tag Git `vMAJOR.MINOR.PATCH` (setuptools-scm): uma build em uma tag sem alterações reporta `MAJOR.MINOR.PATCH`; sem tag, `0.0.1.devN+g<commit>`. Uma árvore sem metadados Git (por exemplo, um zip de código-fonte) cai no fallback `0.0.0`, que o smoke rejeita. O workflow de release confere que a versão da wheel é a da tag ([versioning.md](versioning.md)).

## Linha de comando

O entry point `contextmap` e `python -m contextmap` pertencem à capability `runtime`, que ainda não está em `origin/dev` (branch `milestone/runtime-configuration`). Ao integrá-la, o `pyproject.toml` deve declarar `[project.scripts] contextmap = "contextmap.runtime.cli:main"`. Os testes de `tests/packaging/test_release_metadata.py` já estão preparados: ficam pulados enquanto o módulo não existe e passam a exigir a declaração assim que ele existir.

## Erros de dependência opcional

- Adapters ROS sem `rosbags`: `ModuleNotFoundError: No module named 'rosbags'` ao importar `ros1_bag` ou `ros2_bag`. Instale `contextmap[ros1]` (ou `[ros2]`).
- DINOv2, DINOv3 e CLIP sem o extra: `DinoV2DependencyError`, `DinoV3DependencyError` e `ClipDependencyError` listam os módulos ausentes (`torch`, `transformers`, `Pillow` e, com `transformers` 5.x, `torchvision`). Instale `contextmap[vision]`.
- AlphaCLIP: `AlphaClipDependencyError` pede `torch`, `alpha_clip` e `Pillow`; o `alpha_clip` não vem de extra (tabela acima).
- SAM 3 em `float16`/`bfloat16` sem `torch`: `RuntimeError` pedindo o `torch`.
- Não há fallback para outro backend quando o selecionado falha ([AGENTS.md](../AGENTS.md)).
