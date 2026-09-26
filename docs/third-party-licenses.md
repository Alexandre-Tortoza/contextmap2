# Licenças de terceiros e avisos de modelo

Este documento registra a licença **declarada** de cada dependência e modelo que o ContextMap2 pode usar, e o que o repositório redistribui. Não é aconselhamento jurídico e não substitui a leitura do texto de cada licença.

## Como foi levantado

Em 2026-09-21, por consulta somente leitura a metadados públicos, sem baixar pesos nem executar código de terceiros: API JSON do PyPI, API do GitHub sem token (campo `license` do repositório) e API do Hugging Face sem token (`cardData` do modelo). "Não declarada" significa que o metadado não traz licença, não que não exista. Termos de uso de datasets e dos dados de treino dos modelos **não** foram auditados.

## O que este repositório redistribui

A wheel e o sdist contêm apenas código, documentação e testes deste repositório, sob **AGPL-3.0-only** (`LICENSE`); a wheel tem o pacote `contextmap` e o arquivo de licença. Nenhum peso de modelo, dataset, bag, checkpoint ou runtime de terceiros é empacotado, e `.gitignore` e `tests/packaging/test_repository_hygiene.py` impedem que isso aconteça por engano. Modelos e dados são obtidos pelo usuário, de cada fornecedor, sob os termos dele.

O AGPL-3.0 é uma licença de copyleft de rede. Quem oferecer o ContextMap2 (ou uma versão modificada) como serviço acessível por rede deve ler os termos.

## Dependências instaláveis pelo pacote

| Componente | Onde entra | Licença declarada | Fonte da verificação |
| --- | --- | --- | --- |
| NumPy (`numpy<2.4`) | base | BSD-3-Clause, com componentes embutidos; a série 2.5.x declara `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | PyPI (2.5.3; o intervalo fixado não foi consultado versão a versão) |
| rosbags | extras `ros1`, `ros2` | Apache-2.0 | PyPI (0.11.5) |
| torch | extra `vision` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | PyPI (2.14.0) |
| torchvision | extra `vision` | BSD | PyPI (0.29.0) |
| transformers | extra `vision` | Apache-2.0 | PyPI (5.17.0) |
| Pillow | extra `vision` | MIT-CMU | PyPI (12.3.0) |
| setuptools, setuptools-scm | build | MIT | PyPI (84.0.0, 10.2.3) |

Os wheels padrão de `torch` para Linux dependem de pacotes `nvidia-*` (CUDA, cuDNN) com termos próprios da NVIDIA, que não foram auditados aqui.

## Runtimes que o usuário instala por fora

| Componente | Licença declarada | Observações |
| --- | --- | --- |
| SAM 2 (código, `facebookresearch/sam2`) | Apache-2.0 | não está no PyPI; a validação usou um commit do repositório |
| SAM 2.1 (`facebook/sam2.1-hiera-large`) | Apache-2.0 | pesos abertos |
| SAM 3 (código, `facebookresearch/sam3`) | "SAM License" (o GitHub classifica como "Other") | o pacote `sam3` do PyPI tem o texto da SAM License mas um classifier `MIT License`: **não** confiar no classifier; ler o texto |
| SAM 3 (`facebook/sam3`) | "other", acesso *gated* manual | exige aceitar os termos da Meta no Hugging Face |
| DINOv2 (`facebook/dinov2-base`) | Apache-2.0 | pesos abertos |
| DINOv3 (`facebook/dinov3-*`) | "other" (`dinov3-license`), acesso *gated* manual | exige aceitar os termos; sem execução real registrada no repositório |
| CLIP (`openai/clip-vit-*`) | não declarada no card do Hugging Face; o repositório `openai/CLIP` declara MIT | confirmar o termo dos pesos antes de redistribuir |
| AlphaCLIP (código, `SunzeY/AlphaCLIP`) | Apache-2.0 | não está no PyPI; origem e licença **dos checkpoints** não verificadas, e nenhum foi baixado |
| Qwen2.5-VL-7B, Qwen3-VL-2B/4B | Apache-2.0 | |
| Qwen2.5-VL-3B | `qwen-research` (licença própria, não Apache-2.0) | ler os termos antes de qualquer uso além de pesquisa |
| Florence-2 (`microsoft/Florence-2-large`) | MIT | |
| Gemini | API remota, termos de serviço do Google | sem pesos; o cliente `google-genai` é Apache-2.0. Enviar frames a um serviço externo é uma transferência de dados que exige decisão explícita do dono do dado; a CI nunca chama a API |
| Pointcept (código) e `Pointcept/PointTransformerV3` (pesos) | MIT | os checkpoints são de segmentação supervisionada; a licença dos datasets de treino não foi verificada |
| spconv (`spconv-cu126`) | Apache-2.0 | wheels por versão de CUDA |
| FAST-LIO (`hku-mars/FAST_LIO`) | GPL-2.0 | executado como processo externo (normalmente em container); não é importado, ligado nem empacotado. Distribuir uma imagem que junte o FAST-LIO a este código exige revisão jurídica |

## Dados

O dataset `corridor-02` não é versionado nem redistribuído, e seus termos de redistribuição não foram verificados. Exemplos e artefatos de demonstração publicados com uma release devem ser sintéticos ou ter licença explícita; nenhum resultado sintético é apresentado como desempenho em dados reais.

## Manutenção

Ao adicionar um extra ou um backend, registre aqui a licença declarada, a fonte da verificação e a data. Isso vale também para uma nova versão maior de uma dependência cuja licença mudou de expressão.
