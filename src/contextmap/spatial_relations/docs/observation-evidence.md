# Evidência de observação

Visual Perception ou um raciocínio de linguagem podem afirmar algo relacional sobre a cena ("o caixote está sobre a mesa"). É um sinal útil, mas **não é uma medição**: erra de formas que a geometria não erra, nomeia coisas em termos do upstream e vem de um modelo. Por isso entra como **um canal próprio** (`RelationEvidenceChannel.OBSERVATION`), separado da geometria, e **nunca vira uma relação por si só**. Nenhuma inferência de VLM é executada aqui: só há referências ao que o upstream já produziu.

Nada a montante define uma afirmação relacional (`visual_perception` não tem esse tipo), então esta capability define o menor contrato que a carrega como evidência.

## Contrato

- `UpstreamStatementRef` (`source_run_id`, `statement_id`, `physical_observation_id`, `producer`): a afirmação **exata** por identidade. `producer` (backend, modelo e versão do prompt, como texto opaco) mantém distinguíveis afirmações de produtores diferentes.
- `EndpointLink` (`upstream_ref`, `entity_ref`, `linked_through`): o vínculo **explícito** de uma ponta da afirmação a uma entidade resolvida. `upstream_ref` é o que a afirmação nomeou (uma região, uma claim), `entity_ref` é a `ResolvedEntityReference` e `linked_through` é a evidência por onde o vínculo passou (por exemplo a evidência fundida ou a observação espacial que liga os dois). Sem `linked_through` o vínculo é um palpite e é recusado.
- `ObservationRelationStatement` (`source`, `subject`, `predicate_text`, `object`, `polarity`): o texto do predicado **exatamente como o upstream o disse** e se a afirmação o **afirma** ou o **nega** (`StatementPolarity`). Ela não relaciona uma entidade consigo mesma.

Como a capability não importa `visual_perception` nem `ingestion` (ver `tests/architecture/test_boundaries.py`), os identificadores upstream são textos opacos; quem integra os runs preenche os vínculos.

## De afirmação a evidência

`observation_evidence_from_statements(statements, entities=...)` devolve `ObservationEvidenceResult(evidence, unmapped_statements)`:

- **Vínculo só explícito.** Toda ponta precisa ser uma entidade do conjunto de entidades resolvidas selecionado; caso contrário `UnlinkedEndpointError` (nomeando a ponta). Pontas de artifacts de resolução diferentes, ou um conjunto selecionado que mistura artifacts, levantam `IncompatibleLineageError`. Nada é aproximado.
- **Sem vocabulário escondido.** O texto do predicado só é mapeado quando é exatamente um nome canônico (`canonical_predicate`: `"next to"`, `"NEXT_TO"` e `"next-to"` são `NEXT_TO`). Sinônimos e frases parecidas (`"beside"`, `"near"`, `"under"`, `"resting on"`) **não** são mapeados, porque seria uma expansão silenciosa da ontologia: a afirmação vai para `unmapped_statements` (neutra: nenhuma evidência resulta dela).
- **Direção canônica.** Uma afirmação numa forma derivada (`"below"`) é registrada no sentido avaliado do inverso com as pontas trocadas (`b below a` é `a above b`), e uma simétrica na ordem canônica das duas referências, para cair sobre o candidato que a geometria avaliou. A polaridade se mantém: negar `below` nega `above` do outro sentido.
- **Fusão por candidato.** As afirmações sobre o mesmo candidato viram **um** registro: todas afirmam é `SUPPORTS`, todas negam é `CONFLICTS` e uma mistura é `AMBIGUOUS` com a ressalva `CONFLICTING_STATEMENTS`. As medições são as contagens `asserting_statements` e `denying_statements`; as afirmações exatas (run, id, observação, produtor e vínculos) ficam no registro, ordenadas.

`RelationEvidence` do canal `OBSERVATION` **exige** `statements` (e recusa geometria); os canais medidos **exigem** geometria (e recusam `statements`). A proveniência (`upstream-statements-v1`, versão da taxonomia) não tem frame nem mapa geométrico: afirmações não têm coordenadas.

## Reconciliação: nunca sobrepor a geometria

A [política de decisão](decision.md) lê só os canais medidos (geometria e contato) para decidir. A observação é **corroborante**:

| Situação | Resultado |
| --- | --- |
| só observação, sem evidência medida que decidiu | `UNRESOLVED` (`INSUFFICIENT_EVIDENCE`): nenhuma quantidade de afirmações estabelece uma relação |
| medida sustenta, observação afirma | `SUPPORTED`; a observação fica registrada (`ignored`), sem mudar nada |
| medida sustenta, observação **nega** | `SUPPORTED`; a observação fica registrada com `contradicts_relation=True` |
| medida contradiz, observação afirma | `REJECTED`; a observação fica registrada com `contradicts_relation=True` |
| medida ambígua ou indisponível, observação afirma | `UNRESOLVED`: a observação não desempata |
| nenhuma observação | decide exatamente igual: a ausência é **neutra**, não negativa |

Assim o conflito entre canais é **preservado** (a evidência de observação continua na relação e na decisão, com a contradição marcada), mas uma afirmação do modelo nunca rebaixa uma relação medida nem promove uma não medida.

## Limites conhecidos

- O vínculo entre o que o upstream nomeou e uma entidade resolvida é uma **entrada**: esta milestone não o deriva. A ligação automática (afirmação → claim/região → evidência fundida → entidade → entidade resolvida) precisa do contrato de `ResolvedEntity` de Entity Resolution e fica para a integração.
- A fusão de várias afirmações é a mais simples possível (unânime ou ambígua); não há peso por produtor nem por qualidade.
- Sem execução real: os testes são de contrato com afirmações sintéticas; nenhuma afirmação de VLM real foi usada.
