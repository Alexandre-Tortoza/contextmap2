# Evidência de observação

Visual Perception ou um raciocínio de linguagem podem afirmar algo relacional sobre a cena ("o caixote está sobre a mesa"). É um sinal útil, mas **não é uma medição**: erra de formas que a geometria não erra, nomeia coisas em termos do upstream e vem de um modelo. Por isso entra como **um canal próprio** (`RelationEvidenceChannel.OBSERVATION`), separado da geometria, e **nunca vira uma relação por si só**. Nenhuma inferência de VLM é executada aqui: só há referências ao que o upstream já produziu.

Nada a montante define uma afirmação relacional (`visual_perception` não tem esse tipo), então esta capability define o menor contrato que a carrega como evidência.

## Contrato

- `UpstreamStatementRef` (`source_run_id`, `statement_id`, `physical_observation_id`, `producer`): a afirmação **exata** por identidade. `producer` (backend, modelo e versão do prompt, como texto opaco) mantém distinguíveis afirmações de produtores diferentes.
- `EndpointLink` (`upstream_ref`, `entity_ref`, `linked_through`): o vínculo **explícito** de uma ponta da afirmação a uma entidade resolvida. `upstream_ref` é o que a afirmação nomeou (uma região, uma claim), `entity_ref` é a `ResolvedEntityReference` e `linked_through` é a evidência por onde o vínculo passou (por exemplo a evidência fundida ou a observação espacial que liga os dois). Sem `linked_through` o vínculo é um palpite e é recusado.
- `ObservationRelationStatement` (`source`, `subject`, `predicate_text`, `object`, `polarity`): o texto do predicado **exatamente como o upstream o disse** e se a afirmação o **afirma** ou o **nega** (`StatementPolarity`). Ela não relaciona uma entidade consigo mesma.

Como a capability não importa `visual_perception` nem `ingestion` (ver `tests/architecture/test_boundaries.py`), os identificadores upstream são textos opacos.

## Vínculo a partir de um run de Entity Resolution

Uma afirmação upstream nomeia coisas como a percepção as viu, isto é, **regiões de um frame projetadas no mapa como observações espaciais**; ela não sabe em qual entidade resolvida elas foram parar. Isso está em Entity Resolution: a entidade resolvida traz na sua evidência as observações espaciais de **todos** os membros e lista os membros que fundiu; o leitor do run responde quais entidades resolvidas uma observação espacial sustenta (`EntityResolutionRunReader.resolved_of_spatial_observation`), devolvendo nenhuma ou várias e **nunca escolhendo uma**. `link_statements(afirmações, resolution=leitor)` deriva o vínculo só dessa resposta:

- `UpstreamRelationStatement` (`source`, `subject_spatial_observation_id`, `predicate_text`, `object_spatial_observation_id`, `polarity`) é a afirmação como o upstream a fez, ainda sem entidade;
- uma ponta é ligada à **única** entidade resolvida que o leitor diz que a sua observação espacial sustenta, e o `EndpointLink` resultante registra a observação, a entidade resolvida e os membros (`linked_through`: "spatial observation … is in the evidence of resolved entity … (members: …)");
- afirmações de **membros diferentes** da mesma entidade fundida caem na mesma entidade resolvida e, depois, viram um só registro de evidência;
- uma observação que **nenhuma** entidade tem (`NOT_IN_ANY_ENTITY`), que **várias** têm (`IN_SEVERAL_ENTITIES`, ponta ambígua) ou duas pontas na **mesma** entidade (`BOTH_ENDS_IN_ONE_ENTITY`, que não é uma relação entre duas entidades) **não** são ligadas: vão para `unlinked` com a razão e os identificadores, nunca por proximidade, label ou palpite.

O resultado (`LinkedStatements(linked, unlinked)`, em ordem canônica e independente da ordem de entrada) alimenta `observation_evidence_from_statements`. Uma afirmação sem vínculo é neutra: não produz evidência, e fica listada para que a perda seja visível.

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

- O vínculo usa **observações espaciais** como nome upstream da ponta. Quem transforma a região de um frame numa observação espacial (a projeção de Sensor Association) fica a montante desta capability, que não importa `sensor_association`.
- A fusão de várias afirmações é a mais simples possível (unânime ou ambígua); não há peso por produtor nem por qualidade.
- Sem execução real: os testes são de contrato com afirmações sintéticas; nenhuma afirmação de VLM real foi usada.
