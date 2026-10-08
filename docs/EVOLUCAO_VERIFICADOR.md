# Registro de evolução do verificador

Atualizado em 08/10/2026. Estado: marco 1 implementado e validado localmente; marcos 2/3 com protótipo funcional e coleta/API reais; avaliação factual independente e homologação pendentes. Nenhum deploy ou promoção de modelo foi realizado.

Contrato do frontend preservado: /analisar-url está ativo; /v2/analisar-url
retorna os mesmos campos obrigatórios com evidências separadas em verificacao.
Guia atual: [contrato da API para o frontend](CONTRATO_FRONTEND_API.md).

Especificação: [prompt mestre](PROMPT_MESTRE_EVOLUCAO.md).
Referência inicial: [investigação técnica](INVESTIGACAO_MODELO_2026-10-08.md).
Resultados preservados: [auditoria](../reports/audit_2026-10-08/metrics.json).

| Marco | Estado | Evidência de conclusão necessária |
| --- | --- | --- |
| Diagnóstico inicial | Concluído | Relatório, scores reais, matriz e script de reprodução |
| 1. Base confiável | Concluído localmente | Contratos, representação preservada, pipeline candidato consistente, splits/hashes auditáveis, instalação limpa e testes |
| 2. Evidências atuais | Parcial; marco ativo | Extração, proveniência e duas fontes recentes reais funcionam; busca/recuperação independente e cobertura semântica ainda não validadas |
| 3. Parecer e integração | Parcial | v2, regras/abstenção, cache/limites e casos controlados exercitados; robustez factual externa e operação distribuída pendentes |
| 4. Avaliação independente | Pendente | Corpus recente anotado sem acesso à previsão, métricas e intervalos reproduzíveis |
| 5. Homologação e operação | Pendente | Carga, monitoramento, promoção e rollback comprovados |

## Evidências iniciais — anteriores à implementação, preservadas

- Artefato atual: TF-IDF + LinearSVC calibrado; mDeBERTa para tom. Ainda não há busca factual na aplicação.
- Holdout de títulos: acurácia 90,33%; recall fake 80,73%; 218 falsos negativos e cinco falsos positivos. Artigos correspondentes foram usados na calibração; não é teste independente da API.
- Testes controlados: pares com alteração de número ou remoção de negação geraram a mesma entrada linear e o mesmo score factual.
- Latência local híbrida p95: 2,26 s, sem scraping, evidências, banco ou concorrência. Não representa o futuro fluxo completo.
- Startup real funcionou no diagnóstico. Teste de health existente falhou. Container e staging remoto não foram validados.

## Decisões adotadas e metas ainda propostas

- Conclusões por alegação com evidências e referência temporal; admitir evidência insuficiente e conflito.
- Score lexical e viés permanecem auxiliares. Probabilidade factual depende de avaliação apropriada.
- Desenvolvimento começa com os recursos locais disponíveis e interfaces de fontes; provedor de busca, eventual LLM e orçamento ainda não foram escolhidos.
- Avaliação recente em português é necessária mesmo que se use um benchmark estrangeiro como referência metodológica.
- Modelos candidatos são versionados separadamente; nenhum modelo novo será promovido por este registro.
- Metas de qualidade do prompt são propostas e precisam ser ratificadas antes do teste final.

## Próxima entrega executável — marco 2

Avaliar recuperação de fontes independentes reais para as alegações, definir instância/provedor de busca e melhorar extração/contexto com casos atuais. Preparar material para anotação cega em português sem misturar notícias usadas no desenvolvimento com o teste final. A camada inicial é lexical e conservadora, não um modelo factual robusto.

Dependências externas: instância real de busca/política de fontes; dois revisores e adjudicação; ratificação de metas/custo de erros; SLA e orçamento; Docker/staging. Ainda é possível avançar localmente em extração, fixtures, ferramentas de anotação e monitoramento. Não contratar/publicar/promover automaticamente.

## Registro de execução

Ao avançar, acrescente uma entrada com data, marco, resultado, arquivos relevantes, comandos executados, saída resumida, métricas medidas, decisões, limitações e próxima ação. Marque um marco concluído somente depois de apresentar evidências de seus critérios.

- 08/10/2026: investigação e artefatos diagnósticos concluídos; prompt mestre e registro de progresso preparados. Sem implementação ou retreino da evolução nesta etapa.

### 08/10/2026 — entrega local implementada

Detalhes: [entrega técnica](ENTREGA_EVOLUCAO_2026-10-08.md).
Evidências: [validação](../reports/evolution_2026-10-08/validation.json),
[coleta/API reais](../reports/evolution_2026-10-08/live_retrieval.json),
[testes do ambiente original](../reports/evolution_2026-10-08/pytest-original.xml),
[testes da instalação limpa](../reports/evolution_2026-10-08/pytest-clean.xml).

Entregas:

- NewsDocument/alegações/evidências/pareceres com datas, hashes e semântica explícita. facts-v2 preserva informação; a limpeza legada permanece isolada para os pesos originais.
- Treino TF-IDF + LinearSVC + sigmoid, título/corpo alinhados, agrupamento/deduplicação e três splits temporais. 963 registros sem data e oito de fronteira expurgados. Limitação do corpo transformado registrada; não foi recuperado texto original perdido.
- Candidatos candidate-20261008T194332550409Z e candidate-20261008T195757552426Z em models/candidates; nenhum promovido. Segundo acrescenta tokenização de percentuais/moedas. Ambos têm manifesto/modelo/splits/previsões e métricas.
- Último teste histórico: n=2.242; matriz [[905,63],[64,1210]], linhas/colunas FAKE,REAL; acurácia 94,34%, precisão fake 93,40%, recall fake 93,49%, F1 fake 93,44%, FPR 5,02%, FNR 6,51%, Brier 0,04347. Não cumpre metas propostas de precisão/FPR e não mede qualidade factual atual. Holdout histórico observado durante desenvolvimento, não teste final cego.
- XAI corrigido pela direção do calibrador, domínio verificado por hostname/sufixo, readiness sem fallback factual zero-shot. Tom opcional/offline com revisão fixa; startup não baixa recursos.
- POST /v2/verificar com conclusões por alegação, confiança factual null, fontes rastreáveis, abstenção temporal/semântica, conflitos e deduplicação. SearxNG configurável; nenhuma instância de busca real homologada. Duas origens para decisão unilateral é política conservadora de desenvolvimento, não critério factual validado.
- URLs HTTPS públicas, DNS/IP pinning, redirecionamentos e limites; corpo HTTP limitado antes do JSON; fontes/alegações/concorrência limitadas. Cache v2 sincronizado, por conteúdo/fontes/versão/referência e TTL; v1 depreciada e cache legado versionado com TTL.
- Retreino por votos foi desativado sem leitura/escrita do banco: feedback não revisado não pode contaminar datasets. Monitor Evidently legado continua pendente de modernização.
- Lock e Dockerfile alinhados, tom pesado em grupo opcional, .dockerignore. Novo venv sem Torch/Transformers instalado a partir do lock; stopwords já disponíveis na máquina. Referências originais verificadas por SHA-256 e preservadas.

Comandos e resultados:

~~~bash
.venv/bin/python -m ml.train
.venv/bin/python -m scripts.validate_evolution --candidate models/candidates/candidate-20261008T195757552426Z
.venv/bin/python -m pytest -q --junitxml=reports/evolution_2026-10-08/pytest-original.xml
.venv-validation/bin/python -m pytest -q --junitxml=reports/evolution_2026-10-08/pytest-clean.xml
.venv-validation/bin/python -m pip check
.venv/bin/poetry check --lock
.venv/bin/python -m scripts.live_verification
docker info
~~~

O treino foi executado duas vezes, sempre gerando novo candidato. Validação
recalculou a matriz, confirmou seis hashes de referência e gerou dez casos
explicitamente sintéticos com score textual separado de confiança factual.
Instalação limpa via Poetry 2.5.1 --only main,dev --no-root concluída; pip check
sem dependências quebradas. 83 testes de regressão/contratos/API/segurança passaram
em cada um dos dois venvs, com zero falhas/erros nos XMLs acima. Warnings de bibliotecas
legadas/deprecações permanecem visíveis. Python validado: 3.14.6/macOS ARM64.

A leitura real exigiu acesso de rede autorizado. Falhas iniciais de DNS restrito
e TLS local foram diagnosticadas; bundle de certificados foi incorporado sem
desabilitar verificação. A coleta detectou datas erradas de notícias relacionadas:
campos Publicado/Atualizado passaram a ter prioridade, com precisão/proveniência
registradas, e datas inferidas ambíguas não autorizam decisão.

Duas notícias oficiais de Santos, publicadas em 06/10 e 07/10/2026, foram baixadas
e submetidas à v2 local. HTTP 200 em ambas; parecer inconclusivo por haver apenas
uma origem por alegação. Respostas locais: 178,551 ms e 200,528 ms, após coleta
inicial separada de 3.584,109 ms e 534,828 ms. n=2, sem carga/TCP público/banco/
busca automática: não é SLA nem avaliação de acurácia. Confiança factual e métricas
de qualidade recente permanecem indisponíveis. Não houve revisores humanos.

docker info encontrou cliente, mas nenhum socket/daemon ativo: imagem Linux,
runtime Compose, migrações, carga, staging e custos continuam não validados.
Nenhum banco de produção foi alterado pelos testes, nem serviço pago contratado.

Próxima ação: retomar marco 2 com recuperação independente/contexto, preparar
anotação cega e só depois abrir validação factual e homologação. Artefatos
existentes e candidatos anteriores devem continuar preservados.

### 08/10/2026 — compatibilidade obrigatória com o frontend

Solicitação: manter a saída original de /analisar-url como base obrigatória,
sem exigir que o frontend interprete o JSON técnico de alegações/metadados.

Entregas:

- /analisar-url novamente apresentado como endpoint ativo, sem deprecated. Entrada mínima url e os campos originais de sucesso preservados.
- /v2/analisar-url é a nova fachada por URL: a raiz mantém url, titulo, prob_fake, taxa_confiabilidade, bias_label, bias_score, janela_analise, três listas e modelos_usados; todas as chaves originais estão required no OpenAPI. Tipos nullable anteriores foram preservados, sem fabricar valores.
- Nova informação factual somente em verificacao; corpo completo/metadados do documento e scores auxiliares não são duplicados. A resposta de /v2/verificar continua intacta como contrato de auditoria detalhada.
- Uma função de apresentação compartilhada evita divergência entre as saídas. A v2 usa o mesmo classificador servido e não transforma parecer refutado/conflitante em prob_fake=1 nem em taxa de confiança artificial.
- Entrada da fachada aceita só URL como campo obrigatório. Backend extrai título/corpo/datas; busca configurada é tentada por padrão (pode ser desligada com search=false). Sem provedor informa search_not_configured e admite insuficiência. Busca ignora trechos não verificáveis; nenhum serviço foi configurado ou chamado nos testes.
- Swagger tem exemplos mínimos por URL nas duas fachadas e exemplos separados por URL/texto em /v2/verificar; não sugere document e url juntos nem evidence_urls fictícias.
- Campo obrigatório numérico exige classificador real: ausência produz HTTP 503, não valores preenchidos com zero. Obrigações da resposta valem para sucesso HTTP 200; erros conservam detail.
- Guia [CONTRATO_FRONTEND_API.md](CONTRATO_FRONTEND_API.md) e README atualizados. Frontend existente pode continuar usando /analisar-url. A nova fachada não grava no banco legado; migração de /votar para esse fluxo exige persistência específica e ainda não foi implementada.

Validação local:

~~~bash
.venv/bin/python -m pytest -q --junitxml=reports/evolution_2026-10-08/pytest-frontend-original.xml
.venv-validation/bin/python -m pytest -q --junitxml=reports/evolution_2026-10-08/pytest-frontend-clean.xml
git diff --check
~~~

Suíte completa: 97 testes passaram em cada venv, com zero falhas/erros.
Relatórios novos preservam os XMLs da primeira entrega: [ambiente original](../reports/evolution_2026-10-08/pytest-frontend-original.xml)
e [ambiente limpo](../reports/evolution_2026-10-08/pytest-frontend-clean.xml).
14 testes específicos cobrem entrada mínima, mesma saída v1/v2 em todos os
campos originais, required/nullable/OpenAPI, Swagger, validação antes de extração,
SSRF, busca automática/opção de desligar, classificador ausente, persistência
legada e preservação do score diante de refutação/conflito. Um teste usa os
pesos reais com extração simulada; os demais são controlados. Não houve acesso
a notícia real/provedor nesta entrega nem medição nova de acurácia factual.
git diff --check sem problemas. Seis hashes da auditoria e os hashes dos dois
candidatos foram reconfirmados; sem treino, substituição de artefato, alteração
do banco de produção ou deploy. Warnings de dependências continuam visíveis.

Próxima ação: frontend manter o contrato original ou adotar a fachada v2;
reiniciar/atualizar /docs se o servidor não usa reload. Retomar recuperação
independente e validação humana do marco 2; modelo/provedor não foram promovidos
por essa melhoria de integração.
