# Entrega técnica — evolução em 08/10/2026

## Parecer

Atualização posterior da integração: /analisar-url foi mantido como endpoint
ativo e /v2/analisar-url acrescenta evidências sem mudar os campos originais.
Consulte o [contrato atual do frontend](CONTRATO_FRONTEND_API.md) e o registro.
Os resultados abaixo documentam a primeira entrega, antes desse ajuste de API.

Marco 1 implementado e validado localmente. Marcos 2 e 3 têm um protótipo
funcional, com integração real de duas fontes recentes, mas não estão aceitos
como verificação factual robusta. Marcos 4 e 5 continuam pendentes.
Não houve deploy, promoção de candidato ou alteração dos dados/modelos
de referência. O sistema não está aprovado para publicação em produção.

O progresso está no [registro](EVOLUCAO_VERIFICADOR.md).
Os resultados completos estão em
[validation.json](../reports/evolution_2026-10-08/validation.json),
[live_retrieval.json](../reports/evolution_2026-10-08/live_retrieval.json)
e nos relatórios JUnit em reports/evolution_2026-10-08.

## 1. Base confiável e modelo candidato

- Contrato NewsDocument: título, corpo, português, origem, publicação,
  atualização, evento, acesso e procedência das datas.
- Representação facts-v2 mantém negações, números, percentuais, unidades e
  datas. O preprocessador integra o próprio pipeline; treino e inferência
  candidata recebem a mesma composição de título/corpo.
- O corpus FakeRecogna local não oferece corpos originais íntegros: a coluna
  Noticia já está transformada. Informação anteriormente eliminada não foi
  reconstruída nem alegada como recuperada.
- Agrupamento antes do split por corpo/URL exatos e títulos semelhantes;
  grupos que atravessam fronteiras temporais são expurgados. A heurística não
  comprova independência de todas as narrativas: revisão semântica permanece
  uma limitação.
- Rótulos candidatos: 0=FAKE, 1=REAL. A calibração usa somente o split de
  calibração; não usa artigos do teste.
- Foram gerados dois candidatos, preservados separadamente. O segundo
  acrescenta unidades percentuais/monetárias à tokenização; não foi escolhido
  por um teste final recente. O holdout histórico foi observado durante o
  desenvolvimento, portanto não equivale à avaliação cega final.

Candidato de desenvolvimento atual:
models/candidates/candidate-20261008T195757552426Z.
Há manifesto com SHA-256 do corpus/modelo, versões de dependências e metadados
dos splits. CandidateClassifier verifica hash, representação e rótulos antes
de carregar um artefato local confiável; não é usado automaticamente na API.

| Partição | Artigos | Intervalo | Fake / Real |
| --- | ---: | --- | --- |
| Treino | 5.159 | 09/02/2012–22/07/2020 | 2.862 / 2.297 |
| Calibração | 3.530 | 23/07/2020–08/02/2021 | 1.373 / 2.157 |
| Teste histórico | 2.242 | 09/02/2021–07/09/2021 | 968 / 1.274 |

Foram excluídos 963 registros sem data interpretável e oito registros de
grupos na fronteira. As proporções temporais reais foram mantidas, sem
sobreamostragem antes do split. LinearSVC usa C=1, sem custo de erro
customizado; a classificação candidata mantém threshold 0,5 da calibração.
Nenhuma meta de erro em produção foi ratificada.

### Métricas medidas — exclusivamente históricas

| Métrica | Resultado |
| --- | ---: |
| Acurácia | 94,34% |
| Precisão FAKE | 93,40% |
| Recall FAKE | 93,49% |
| F1 FAKE | 93,44% |
| F1 macro | 94,23% |
| FPR: real classificada como fake | 5,02% |
| FNR: fake classificada como real | 6,51% |
| Brier no teste histórico | 0,04347 |

Matriz: linhas verdade de referência; colunas previsão.

| Referência / Previsão | FAKE | REAL |
| --- | ---: | ---: |
| FAKE | 905 | 63 |
| REAL | 64 | 1.210 |

A matriz foi recalculada a partir de test_predictions.csv. Não é comparável
diretamente com os 90,33% auditados anteriormente: populações e protocolo
mudaram. Não mede veracidade atual, nem cumpre as metas propostas de
precisão ≥95% e FPR ≤2%. Brier histórico não calibra confiança factual.

## 2. Referência legada preservada e corrigida

A probabilidade histórica não foi invertida por suposição. O índice FAKE é
localizado em classes_; a explicação deriva a direção do sigmoid calibrador.
Testes verificam contribuições e impedem reconhecimento de gov.br.evil.example
como fonte governamental.

O classificador legado conserva sua limpeza para manter compatibilidade
com os pesos de referência; ela ainda perde informação. As próximas entradas
candidatas não usam essa limpeza. API v1 foi depreciada e explicita que
prob_fake/taxa_confiabilidade não constituem confiança factual.

Sem SVC/stopwords, a prontidão retorna indisponível e não existe fallback
zero-shot que invente uma detecção factual. O modelo de tom é opcional,
com revisão fixa, e carrega somente arquivos locais. Startup não baixa
recursos e não desabilita TLS.

O retreino automático por votos foi desativado: não acessa nem altera o banco
ou o corpus. Feedback continua recebível, mas requer revisão independente
antes de virar dado de treino. O monitor legado em ml/drift_monitor.py ainda
precisa ser modernizado; não foi apresentado como operacional.

## 3. Evidências, parecer e API v2

POST /v2/verificar recebe texto ou URL, fontes opcionais e referência temporal.
Sem provedor, a busca não é simulada. SearxNG pode ser configurado em uma
instância HTTPS escolhida pelo operador; fixtures testam o contrato e URLs.
Nenhuma instância real de busca foi contratada ou homologada.

A extração heurística limita-se a 12 alegações. A comparação é estrita:
mesma estrutura lexical, entidade, números e negação. Diferença de data de
evento não vira refutação numérica. Quantidades aproximadas, linguagem
atribuída, perguntas, sátira explícita e datas relativas não resolvidas
produzem abstenção. Sarcasmo implícito e paráfrases não estão resolvidos.

Uma decisão unilateral exige duas origens; posições incompatíveis são
exibidas como conflito. Domínios registráveis, cópias exatas e cópias próximas
reduzem dupla contagem, mas não demonstram independência editorial.
Fontes primárias não têm confiança automática por reputação. Autoridade
específica e contexto ainda exigem avaliação humana.

Cada evidência inclui URL, trecho, publicação/atualização, acesso, hash,
procedência e relação com a alegação. confidence=null; relevance_score é
similaridade lexical não calibrada. O score textual candidato é mostrado
separadamente no relatório sintético, nunca utilizado para alterar o parecer.

### Dez exemplos controlados

Todos os casos abaixo e suas fontes .example são fixtures sintéticas.
Não são notícias reais nem medição de acurácia factual recente.

| Caso | Parecer por alegação | Score textual fake |
| --- | --- | ---: |
| Corroboração controlada | Sustentada | 0,2626 |
| 82 alterado para 8200 | Refutada | 0,2626 |
| Negação alterada | Refutada | 0,5704 |
| Fontes incompatíveis | Evidências conflitantes | 0,2626 |
| Sem fonte | Evidência insuficiente | 0,2626 |
| Artigo parcial | Sustentada + insuficiente | 0,2107 |
| Título interrogativo/clickbait | Insuficiente | 0,3178 |
| Sátira declarada | Insuficiente | 0,3172 |
| Fonte antiga sem evento explícito | Insuficiente | 0,2302 |
| Republicações idênticas | Insuficiente | 0,2626 |

O score textual pode permanecer igual depois de alterar um número:
o vocabulário aprendido tem cobertura finita. A representação preserva
essa alteração; a comparação com evidências reage a ela. Não se presume
que o modelo textual seja um verificador numérico.

### Coleta real recente

Foram baixadas e submetidas à API local duas publicações oficiais:

- [Plano preventivo da Defesa Civil](https://www.santos.sp.gov.br/?q=noticia/santos-define-orgaos-que-atuarao-no-plano-preventivo-da-defesa-civil): publicação 07/10; atualização 07/10/2026.
- [Equipamentos para Ilha Diana](https://www.santos.sp.gov.br/?q=noticia/ilha-diana-em-santos-recebe-equipamentos-para-atendimento-em-situacoes-de-emergencia): publicação 06/10; atualização 08/10/2026.

Ambas retornaram HTTP 200 e evidência insuficiente: uma origem por alegação.
Foi selecionado um trecho real de cada notícia, sem inventar confirmação
independente. Isto comprova leitura/extração/API reais, não a verdade da
notícia ou generalização do modelo. Não houve anotação humana cega.

A validação revelou e corrigiu dois problemas: certificados TLS ausentes na
configuração padrão local e datas genéricas extraídas de notícias relacionadas.
Agora há bundle de CAs, verificação HTTPS mantida e prioridade para metadados
explícitos/campos Publicado/Atualizado. Datas genéricas ambíguas não autorizam
decisão. Campos visíveis sem timezone são registrados com precisão de dia;
horário de acesso usa UTC real e não é confundido com horário de publicação.

## 4. Segurança, instalação e testes

URLs só aceitam HTTPS público, sem credenciais/portas alternativas. Todos os
IPs DNS são validados, inclusive em redirecionamentos; a conexão é fixada ao
IP validado, preservando SNI e certificado. Há limite de bytes, tipo MIME,
redirecionamentos e tempo, sem descompressão de respostas.

A API limita o corpo antes do parsing JSON a 512.000 bytes; documentos a
100.000 caracteres; até cinco fontes; quatro trabalhos simultâneos por
processo, fila de um segundo e orçamento de coleta de 30 segundos.
DNS tem fila limitada e timeout; threads DNS eventualmente presas permanecem
limitadas. Firewall de saída e rate limiting distribuído são necessários
para produção.

Cache v2 tem TTL de 300 segundos e capacidade de 128 resultados por processo,
com sincronização entre threads. Chave inclui documento, conteúdo/datas das
fontes, versão e referência; as fontes são recuperadas antes da reutilização.
Não é um cache distribuído nem elimina o custo de coleta. Cache legado usa
TTL e versão, sem migração/destruição dos registros existentes.

Lock regenerado com Poetry 2.5.1 e NumPy compatível. Instalação em um venv
separado, somente main/dev, funcionou; pip check não encontrou dependências
quebradas. Os testes passaram também nesse ambiente sem Torch/Transformers.
Os stopwords já existiam na máquina; provisionamento explícito em máquina
nova está descrito no README. Python efetivamente validado: 3.14.6/macOS ARM64.

83 testes passaram em cada um dos dois ambientes, sem falhas. Foram exercitados
contratos, labels, hashes/splits, representação, XAI,
readiness, API legada e v2, extração datada, SSRF, DNS pinning, redirecionamentos,
limites, provedores, cache, abstenção, conflito, cópias e temporalidade.
Warnings de bibliotecas legadas/deprecações permanecem registrados em JUnit;
não foram suprimidos para fingir aprovação.

Dockerfile foi alinhado ao lock/Python, sem dependência do plugin export e
sem download de Hugging Face. .dockerignore evita incluir datasets, ambientes
e candidatos. Build/runtime Linux e migrações não foram validados: docker info
falhou porque o socket/daemon não está disponível. Não há staging/publicação.

## 5. Latência e próximos bloqueios

Nos dois exercícios reais com TestClient e uma fonte, respostas locais
levaram 178,551 ms e 200,528 ms. Coleta inicial separada levou 3.584,109 ms e
534,828 ms. Amostra n=2, rede já aquecida na API, sem TCP público, banco,
busca automática ou concorrência: não é benchmark nem SLA em tempo real.
O relatório também contém latência sintética n=10, com escopo explícito.
CPU atende ao protótipo local; necessidade de GPU, RAM sob carga e custo
cloud não foram medidos. Tom opcional exige recursos adicionais.

Próxima entrega: avaliar recuperação real independente para as mesmas
alegações, melhorar contexto/extração e preparar corpus de anotação cega em
português com dois revisores/adjudicação. Critérios/metas devem ser ratificados
antes de abrir o teste final. Fontes atuais usadas para desenvolvimento
não devem ser recicladas como teste cego.

Para produção ainda faltam corpus revisado, métricas factuais/abstenção com
intervalos, política de fontes, carga, imagem Linux, migrações, autenticação,
monitoramento com rótulos revisados, retreino moderado, registro de promoção/
rollback e aprovação de infraestrutura/SLA. Sem isso, nenhum cronograma
fechado, preço cloud ou aprovação de deploy foi inventado.

