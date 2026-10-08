# Prompt mestre para evolução do verificador de notícias

Copie o bloco abaixo para iniciar o desenvolvimento no repositório. Ele é uma especificação de trabalho proposta; a criação deste documento não significa que as mudanças no modelo já foram implementadas.

```text
Atue como engenheiro de Machine Learning, NLP e backend neste repositório.
Implemente, por marcos verificáveis, a evolução do identificador de fake news
para um sistema de verificação de alegações em notícias atuais em português.

OBJETIVO
Receber URL ou texto, identificar alegações verificáveis, recuperar evidências
relevantes e datadas, comparar cada alegação com essas evidências e retornar
um parecer rastreável. Use sustentada, refutada, evidências conflitantes ou
evidência insuficiente por alegação. Represente temporalidade e parcialidade
sem forçar todo artigo a um rótulo binário. Ausência de evidência não implica
falsidade. Tom sensacionalista, reputação de domínio e score lexical não
substituem checagem factual.

CONTEXTO
Leia docs/INVESTIGACAO_MODELO_2026-10-08.md e
docs/EVOLUCAO_VERIFICADOR.md. Inspecione o código e as mudanças existentes.
O projeto tem FastAPI, extração web, TF-IDF + LinearSVC calibrado,
mDeBERTa zero-shot para tom, cache e componentes parciais de MLOps.
A auditoria identificou treino em títulos e serving em corpos, rótulos
inconsistentes, artigos do holdout usados na calibração, perda de números
e negações, XAI invertido e falta de avaliação recente independente.
Os 90,33% de acurácia auditados não validam desempenho em notícias atuais.
Confirme o estado atual antes de agir: essas evidências têm data e podem
ter sido superadas por mudanças posteriores.

FORMA DE TRABALHO
Mantenha docs/EVOLUCAO_VERIFICADOR.md como registro de marcos, critérios,
decisões, comandos, resultados, limitações e próxima ação. Retome trabalho
já concluído sem refazê-lo. Execute implementação, validação e correção dos
problemas do marco ativo. Não encerre apenas com um plano se houver trabalho
local viável. Resolva escolhas técnicas rotineiras com justificativa.
Quando faltar uma informação essencial, avance nas partes independentes
e explique exatamente qual escolha ou acesso está faltando.

MARCO 1 — BASE CONFIÁVEL
Padronize o contrato de título, corpo, idioma, fonte, datas e rótulos.
Preserve informação de negação, números, unidades e datas. Verifique se o
corpus já foi transformado e se o texto original está disponível; dados
apagados do CSV limpo não podem ser reconstruídos por nova limpeza.
Unifique treino e inferência. Separe treino, validação/calibração e teste
por artigo/narrativa e tempo, com deduplicação. Persista a semântica das
classes: não troque as probabilidades do artefato atual por suposição.
Gere modelos candidatos em caminhos novos, com hashes e metadados, sem
promovê-los sobre o modelo servido. Corrija XAI, prontidão e os testes
essenciais da API; faça a instalação ser reproduzível.
Aceite o marco com comandos reproduzíveis, entrada consistente, splits
auditáveis e testes que detectem regressões de números, negações e rótulos.
Não exija que todo exemplo tenha previsão diferente: informação diferente
deve ser preservada e o comportamento final deve ser avaliado.

MARCO 2 — EVIDÊNCIAS ATUAIS
Implemente coleta de título, corpo e data com procedência; depois extração
de alegações e recuperação de fontes. Priorize fontes primárias adequadas
ao assunto e procure confirmação independente quando possível. Artigos que
apenas copiam uma mesma origem não são evidências independentes.
Registre URL, trecho pertinente, publicação, acesso e vínculo com a alegação.
Distinga data do evento, publicação, atualização e referência temporal.
Crie interfaces de provedores e exemplos locais determinísticos; valide
também recuperação real quando houver acesso. Mock não comprova busca web.
Para começar, use ferramentas locais disponíveis e templates auditáveis.
Se um provedor depender de credenciais ou contratação, prepare a integração
e continue trabalho independente, explicitando a dependência. Um LLM pode
ser comparado posteriormente, mas não é condição inicial nem fonte de fatos.
Aceite o marco quando alegações tiverem evidência recuperada pertinente e
quando indisponibilidade de fonte produzir evidência insuficiente.

MARCO 3 — PARECER E INTEGRAÇÃO
Implemente comparação alegação-evidência com tratamento de quantidade,
negação, entidade e tempo. Avalie regras e modelos semânticos por resultados,
sem presumir que NLI ou geração de texto constitui verificação confiável.
Retorne conclusão por alegação, evidências, justificativa vinculada aos
trechos, limitações, temporalidade e versão do sistema. Confiança só deve
ser chamada de probabilidade se houver avaliação de calibração aplicável;
caso contrário, identifique o score ou retorne confiança indisponível.
Mantenha tom e score lexical como informações auxiliares separadas.
Evolua a API com versão e compatibilidade documentadas. Implemente cache por
conteúdo/versão com expiração, limites de tamanho/tempo e proteção de URLs
contra redes privadas e redirecionamentos indevidos.
Aceite o marco com fluxo completo exercitado, evidência rastreável e casos
inconclusivos, conflitantes, parciais e antigos tratados explicitamente.

MARCO 4 — AVALIAÇÃO INDEPENDENTE
Congele candidato e threshold antes do teste. Crie corpus recente com
notícias das últimas semanas e última semana, anotação sem acesso à predição,
revisão e evidência de referência. Separe textos reais, sintéticos e
históricos. Inclua alegações numéricas, negações, clickbait, sátira e
reutilização de notícias antigas. Não trate votos como ground truth.
Reporte matriz, precisão, recall, F1, FPR/FNR, calibração quando aplicável,
qualidade da recuperação, correção das evidências e taxa de abstenção,
com intervalos, tamanho da amostra e cortes por tema/fonte/período.
Avalie as decisões automáticas e a cobertura sobre todas as entradas.
As metas iniciais propostas são precisão >=95%, recall >=90% e FPR <=2%
nos casos binários aplicáveis; ECE <=0,05 se houver probabilidade calibrada.
Ratifique o custo dos erros e as metas antes de abrir o teste final.
Aceite o marco somente com dados independentes e resultados reproduzíveis;
se faltar anotação humana, registre a lacuna sem fabricar aprovação.

MARCO 5 — HOMOLOGAÇÃO E OPERAÇÃO
Valide imagem, migrações, startup offline, fallback explícito e carga com
extração, evidências, inferência, banco e concorrência. Meça p50/p95/p99,
memória, throughput e custo por análise. Documente o SLA medido.
Implemente monitoramento de dados e desempenho com rótulos revisados,
feedback moderado, registro de versões e rollback. Retreino deve gerar
candidato; promoção exige superar critérios em validação preservada.
Aceite o marco com homologação exercitada e procedimento de implantação
concreto. Deploy público, serviços pagos e promoção do modelo servido
dependem de autorização específica; prepare os artefatos antes de solicitá-la.

ENTREGA EM CADA MARCO
Informe: resultado alcançado; alterações principais; validação e métricas
medidas; pendências; próxima ação. Atualize o registro com evidências.
Distinga implementado, testado localmente, validado externamente e publicado.
Preserve alterações do usuário e dados/modelos originais de referência.
Nunca invente métricas, notícias, URLs, evidências, confiança ou testes.

COMECE AGORA
Confira o estado do repositório e do registro. Selecione o primeiro marco
pendente, detalhe critérios executáveis e implemente sua primeira entrega
com validação. Continue nas partes viáveis e registre qualquer bloqueio real.
```

Para retomar em outra sessão:

```text
Continue a evolução seguindo docs/PROMPT_MESTRE_EVOLUCAO.md.
Leia o registro docs/EVOLUCAO_VERIFICADOR.md e confira o estado do código.
Retome o primeiro marco pendente, implemente e valide as próximas entregas
viáveis, preserve decisões anteriores e atualize o registro com evidências.
```

Para acompanhar sem solicitar mudanças:

```text
Revise o progresso usando docs/EVOLUCAO_VERIFICADOR.md e evidências do código.
Não altere o projeto nesta revisão. Informe marco atual, entregas concluídas,
validações executadas, bloqueios, desempenho em notícias atuais e próxima
ação. Diferencie preparação técnica, validação externa e prontidão para deploy.
```

A divisão entre especificação, marcos e registro de progresso segue o princípio de memória persistida no repositório descrito nas [orientações oficiais do Codex para tarefas longas](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex). A necessidade de avaliar conjuntamente veredito e evidência é também ilustrada pelo [benchmark AVeriTeC](https://aclanthology.org/2024.fever-1.1/); ele é uma referência metodológica, não validação deste projeto ou substituto de corpus recente em português.
