# Política de segurança

## Versões suportadas

O ContextMap2 está em pre-alpha (série `v0.x`, [docs/versioning.md](docs/versioning.md)). Somente a release `v0.x` mais recente e o estado atual de `main` e `dev` recebem correções; não há backports para releases anteriores.

## Reportando uma vulnerabilidade

Não publique credenciais, tokens, datasets privados, dados pessoais ou detalhes exploráveis de segurança em uma issue pública.

Para defeitos comuns que não exponham informações sensíveis, use o template de bug. O canal previsto para problemas sensíveis é o private vulnerability reporting do GitHub, mas em 2026-09-21 ele estava **desabilitado** neste repositório (verificado pela API; ver [docs/repository-settings.md](docs/repository-settings.md)). Até que seja habilitado, abra uma issue pública **sem** detalhes técnicos pedindo um canal privado à pessoa mantenedora (@Alexandre-Tortoza).

## Dados de pesquisa

Não faça commit de datasets brutos, ROS bags, checkpoints de modelos, artefatos gerados contendo informações sensíveis ou secrets.

Entradas e saídas grandes de pesquisa devem ser referenciadas por meio de armazenamento externo documentado e manifests reproduzíveis. O `.gitignore` cobre `workspace/`, `datasets/`, `outputs/`, `runs/`, `artifacts/`, pesos e bags, e `tests/packaging/test_repository_hygiene.py` falha se um arquivo desse tipo, uma credencial ou um caminho local for rastreado.

## Notas de segurança para quem executa o pipeline

- **Checkpoints são código.** Carregar um checkpoint em formato pickle (`torch.load`) executa código arbitrário. Use apenas checkpoints de origem verificada, prefira `safetensors`, fixe a revisão (`revision`) e mantenha os pesos locais (os adapters usam checkpoints locais por default). A origem e a licença de cada modelo estão em [docs/third-party-licenses.md](docs/third-party-licenses.md).
- **Segredos só por variável de ambiente.** Chaves de API (por exemplo, `GEMINI_API_KEY`) nunca entram em arquivos de configuração, manifests, provenance, logs, issues ou pull requests.
- **Dados não saem da máquina sem decisão explícita.** Enviar frames de um dataset a um serviço externo (por exemplo, Gemini) é uma transferência de dados que o dono do dado precisa aprovar; a CI não chama modelos nem APIs.
- **Cadeia de suprimentos dos workflows.** Permissões mínimas por job; o único job com escrita de conteúdo é o `publish` do release, que não usa segredos além do `GITHUB_TOKEN`; workflows acionados por `pull_request_target` nunca fazem checkout de código de PR; as actions são fixadas por versão principal e atualizadas semanalmente pelo Dependabot. As permissões, o uso de `pull_request_target` e a versão única de cada action são verificados por `tests/packaging/test_workflows.py`. O secret scanning e o push protection do GitHub estavam habilitados na verificação de 2026-09-21.
- **Integridade das releases.** Cada release traz `SHA256SUMS` com os checksums da wheel e do sdist.
