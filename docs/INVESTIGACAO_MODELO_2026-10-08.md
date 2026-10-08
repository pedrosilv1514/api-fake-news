# Investigação e diagnóstico do classificador de fake news

Data: 08/10/2026. Repositório inspecionado no commit `117c0ef`. Escopo: revisão do código, notebooks e artefatos existentes; inferência com os dois modelos reais; avaliação dos datasets locais; consulta de fontes recentes; diagnóstico de prontidão para produção. Não houve retreino, substituição de pesos, alteração da API ou escrita no banco da aplicação.

**Parecer: não liberar o modelo atual como verificador automático de veracidade.** O motor executa localmente, mas a avaliação disponível não sustenta generalização para notícias atuais. Há divergência entre título e corpo no treinamento, calibração sobre artigos do holdout, perda de negações e números, explicações com polaridade invertida e componentes de MLOps incompletos. A aplicação é um protótipo integrado que classifica padrões de texto; ainda não confirma alegações contra evidências externas.

## 1. Situação atual do modelo

### Arquitetura e técnicas de NLP

O caminho efetivamente usado pela API é [ml/inference.py](../ml/inference.py):

| Componente | Implementação efetiva | Limite da capacidade |
| --- | --- | --- |
| Classificação de fake news | `models/calibrated_pipeline.joblib`: `CalibratedClassifierCV`, método sigmoid, com um `FrozenEstimator` envolvendo TF-IDF + LinearSVC | Aprende correlações lexicais com os rótulos do corpus; não verifica fatos |
| Representação textual | TF-IDF de unigramas e bigramas; 50.000 features; `max_df=0.85`, `min_df=1` | Não representa explicitamente relações entre entidades, quantidades e datas |
| Classificador linear | `LinearSVC(C=10, max_iter=2000, random_state=42)` | A margem do SVM, por si só, não é probabilidade |
| Viés/tom da escrita | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, zero-shot, dois rótulos semânticos | O score compara os rótulos escolhidos; não foi validado como probabilidade de veracidade ou de viés neste projeto |
| Explicação | `ExplainabilityEngine`, contribuições TF-IDF × coeficiente, com templates | Associação lexical não é evidência factual; a direção das contribuições está errada para este artefato |
| Extração | `httpx` + `trafilatura` + sanitizador | A extração atual usa apenas o corpo na inferência; o título é retornado separadamente |

Apesar do termo “ensemble” na documentação, não há votação, média ou fusão entre os dois modelos para a factualidade. `prob_fake` vem do SVC calibrado; o Transformer produz `bias_label` e `bias_score`. Eles executam sequencialmente dentro de `predict`. O fallback usa o Transformer para “verdadeira/falsa” se o SVC falhar; esse modo também não tem avaliação própria, e a resposta continua listando os dois modelos.

O mDeBERTa é um encoder Transformer de NLI pré-treinado em CC100 e ajustado em MNLI/XNLI; não há evidência de fine-tuning local em notícias brasileiras ou em rótulos de sensacionalismo. Não há LLM generativo customizado, grafo ou sistema de busca factual no caminho de inferência. [Model card do autor](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).

A limpeza em [ml/preprocessing.py](../ml/preprocessing.py) converte para minúsculas, substitui URLs por `urltoken`, remove números e caracteres especiais, remove stopwords NLTK em português e descarta tokens de até dois caracteres. Preserva acentos. Não aplica stemming/lematização nessa função. O baseline antigo definiu também remoção de acentos e stemming RSLP, mas o CSV exportado usa `texto_limpo`, não `texto_processado`. O sanitizador da API já elimina URLs antes da limpeza, de modo que a preservação via `urltoken` não acontece nesse caminho.

O Transformer recebe só os primeiros **1.500 caracteres**, não 1.500 tokens. Isso pode esconder uma ressalva ou desmentido no fim do artigo. O SVC recebe o texto limpo inteiro.

### Dados, balanceamento e origem dos artefatos

A base local é FakeRecogna, com `Titulo`, `Subtitulo`, `Noticia`, `Categoria`, `Data`, `Autor`, `URL` e `Classe`. Os rótulos originais são **0 = falsa, 1 = verdadeira**. A existência pública do corpus foi confirmada, mas as contagens abaixo foram medidas nos arquivos locais, sem presumir equivalência com versões externas. [Corpus publicado pelo grupo Recogna](https://huggingface.co/datasets/recogna-nlp/FakeRecogna).

| Conjunto local | Total | Falsas | Verdadeiras | Utilização observada |
| --- | ---: | ---: | ---: | --- |
| CSV completo, raiz e `data/` | 11.902 | 5.951 | 5.951 | Baseline e calibração atual |
| V2 antes do split | 11.521 | 5.655 | 5.866 | Títulos limpos, sem vazios e duplicatas de título |
| Treino V2 | 9.216 | 4.524 | 4.692 | Ajuste do SVC e GridSearchCV |
| Holdout V2 | 2.305 | 1.131 | 1.174 | Avaliação reproduzida nesta investigação |

O notebook registra 11.903 linhas iniciais, das quais 11.902 têm conteúdo/classe. O corpus completo é balanceado; a V2 fica próxima de 49%/51%. O split é estratificado 80/20 com seed 42. Não foram encontrados SMOTE, oversampling ou pesos de classe no SVC atual. Ainda existe uma duplicata exata em `texto_limpo` no CSV completo. Não há sobreposição exata entre títulos do treino e do holdout, mas não foi comprovada ausência de duplicatas aproximadas, assuntos ou fontes comuns.

As categorias mais frequentes são saúde (4.456) e política (3.951). Entre 10.939 datas interpretáveis como dia/mês/ano, o intervalo observado foi **09/02/2012 a 07/09/2021**; 963 datas não foram interpretadas por essa extração. Esse intervalo é parcial, não uma afirmação sobre todas as datas do corpus. A separação por fontes também merece avaliação: falsas incluem 2.484 registros de boatos.org; verdadeiras incluem 3.437 de noticias.uol.com.br. Isso não prova uso direto do domínio pelo modelo, mas cria risco de aprender o estilo de cada fonte. Textos de checagem e alegações que eles desmentem precisam ter papéis claramente definidos.

Foram identificadas quatro divergências relevantes de proveniência:

1. **A V2 treina em títulos.** No notebook `01_preproc_eda.ipynb`, o seletor procura `noticia` em minúsculas. A coluna real é `Noticia`; o fallback seleciona a primeira coluna, `Titulo`. Os títulos usados no treino têm média de 8,13 palavras limpas, contra 95,62 no CSV completo. A API analisa corpo de notícia. Compartilhar uma função de limpeza não elimina essa diferença.
2. **O notebook recodifica os rótulos.** `pd.factorize` registra `{0: 1.0, 1: 0.0}`. Assim, no SVC base, classe 0 significa verdadeira e classe 1 significa falsa. `classes_=[0,1]` não informa essa semântica. O Streamlit e outros notebooks também usam convenções diferentes.
3. **A calibração usa outro domínio de texto e todos os artigos.** [calibrate_model.py](../calibrate_model.py) congela o SVC e ajusta o sigmoid sobre os 11.902 textos completos com os rótulos originais. Todos os 2.305 títulos de holdout correspondem a artigos presentes nesse corpus. Não é um teste independente do modelo calibrado, mesmo que título e texto completo sejam strings diferentes. A documentação recomenda separar os dados de ajuste do classificador dos dados de calibração. [Scikit-learn: calibração](https://scikit-learn.org/stable/modules/calibration.html).
4. **O treino novo não é o artefato servido.** [ml/train.py](../ml/train.py) define um modelo calibrado em cinco folds e registra no MLflow; a API carrega um caminho fixo em disco. O artefato existente tem um único estimador congelado. Não há evidência de promoção do pipeline novo à API.

A calibração **já reorienta a probabilidade** para os rótulos originais. Portanto, a inversão no treino não implica simplesmente trocar as colunas de `predict_proba` na API: essa troca sem refazer a proveniência pioraria o resultado atual. O problema exige um contrato único de rótulos, treinamento e calibração reproduzíveis.

### Métricas mais recentes, reproduzidas em 08/10/2026

Positivo significa **fake news**, embora seja codificado por 0 no corpus. Precisão, recall e F1 da tabela referem-se à classe falsa; macro-F1 é informado separadamente. O threshold de auditoria é `prob_fake >= 0.50`; a API retorna score, não uma classe explícita.

| Avaliação | Acurácia | Precisão fake | Recall fake | F1 fake | F1 macro | Interpretação |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Artefato servido, holdout de 2.305 títulos | 90,33% | 99,46% | 80,73% | 89,12% | 90,20% | Auditoria reproduzida; artigos incluídos na calibração |
| SVC base, mesmos títulos, rótulos interpretados corretamente | 94,62% | 95,90% | 93,02% | 94,43% | 94,61% | Holdout do classificador base; não mede a API sobre corpos |
| Artefato servido, 11.902 textos da própria calibração | 91,62% | 92,01% | 91,16% | 91,58% | 91,62% | Diagnóstico dentro da amostra de ajuste; não mede generalização |

O SVC base atinge 100% nos títulos do treino e 94,62% no holdout. A diferença sugere examinar overfitting, mas não permite atribuir os erros apenas a ele; domínio de texto, fontes e calibração também influenciam.

Há resultados históricos, que não devem ser apresentados como métricas atuais da API: F1 macro de CV V2 de 93,66%; acurácia do LinearSVC baseline de 97,14%; e 95,88% no JSON de regressão logística. Parte dos valores no exportador da regressão tem defaults fixos, limitando a rastreabilidade desse resumo.

O notebook chamado “validação externa OOD” mostra `[[5017,263],[336,4944]]`, isto é, 94,33% de acurácia em 10.560 exemplos de uma base abstrativa de 52.800 linhas. Porém ele **treina um novo modelo em 80% dessa própria base** e testa nos 20% restantes. A célula de avaliação realmente externa está comentada. A base abstrativa não está disponível aqui para reprodução e seu `.joblib` separado não é o carregado pela API.

### Ambiente atual

Existem notebooks Jupyter, protótipo Streamlit, API FastAPI v0.1.0, SQLite local, migrações Alembic, Dockerfile, Compose com PostgreSQL e GitHub Actions. A inferência real funcionou offline com os pesos Hugging Face já em cache. O startup real, em `TestClient` com lifespan ativo, retornou `/health` HTTP 200 e `ml_ready=true`.

O banco local contém 15 análises e zero votos. O `mlflow.db` contém apenas o experimento Default: zero runs, métricas, modelos registrados e versões. Além disso, o treino aponta para `mlruns.db`, outro nome. Não há evidência local de staging remoto ou serviço publicado. Não foi observado listener na porta 8000 durante a inspeção; o funcionamento demonstrado foi em processo de diagnóstico, não um deployment persistente.

O teste existente `tests/test_health.py` falhou: espera apenas `{"status":"ok"}`, mas recebe também `ml_ready`. Ele não abre o contexto do cliente e, portanto, não valida o startup. A suíte de extração não foi executada nesta investigação: depende de rede e captura `ExtractionError` sem falhar, podendo dar resultado positivo sem extrair notícia alguma. Docker não foi construído porque o daemon local não está em execução; não há validação do contêiner nesta entrega.

## 2. Verificação de funcionamento e acurácia

### Matriz de confusão atualizada

Artefato `models/calibrated_pipeline.joblib`, limpeza da API, 2.305 títulos do holdout V2. Linhas = verdade de referência; colunas = previsão. **A matriz é atualizada e reproduzível, mas não independente da calibração nem representativa da população de notícias atuais.**

| Referência / previsão | Falsa | Verdadeira | Total |
| --- | ---: | ---: | ---: |
| Falsa | 913 (TP) | 218 (FN) | 1.131 |
| Verdadeira | 5 (FP) | 1.169 (TN) | 1.174 |
| Total | 918 | 1.387 | 2.305 |

- Falso positivo: `5 / 1174 = 0,426%` das verdadeiras marcadas como falsas.
- Falso negativo: `218 / 1131 = 19,275%` das falsas liberadas como verdadeiras.
- Precisão fake: `913 / 918 = 99,455%`. Não confundir com recall fake: `913 / 1131 = 80,725%`.

Para comparação, a matriz do SVC base com a semântica dos rótulos corretamente interpretada é `[[1052,79],[45,1129]]`: FPR de 3,83% e FNR de 6,98%. No corpus completo da calibração, o artefato servido produz `[[5425,526],[471,5480]]`: FPR de 7,91% e FNR de 8,84%. Essas diferenças mostram por que o conjunto avaliado precisa acompanhar cada número.

**Qual erro é otimizado?** O GridSearch antigo maximiza F1 macro; o treino novo apenas registra F1 ponderado. Não há função de custo assimétrica, meta explícita para FPR/FNR ou seleção documentada de threshold. O resultado atual nos títulos favorece empiricamente poucos falsos positivos, com perda de recall. Isso não demonstra uma decisão deliberada de produto.

Exemplos de erros existentes podem ser auditados em `title_holdout_predictions.csv`: o título rotulado real “alvarez marsal gostou vídeo moro citando edith piá” recebe `prob_fake≈0,9910`; “bolsonaro pretende extinguir ipva imposto sobre automóveis”, rotulado falso, recebe `prob_fake≈0,0060`. São rótulos históricos do corpus, não novas checagens realizadas aqui.

### Dez exemplos práticos com os dois modelos reais

Foram fixados dez textos antes de executar as previsões. R1, R3 e R5 são resumos redigidos para este diagnóstico a partir de publicações de 06–07/10/2026; R2 e R4 são alterações controladas; R6 é paráfrase de uma alegação desmentida. E7–E9 são edge cases derivados/sintéticos e E10 resume uma notícia histórica rotulada real no corpus. **Não são dez artigos íntegros coletados às cegas.** Os textos integrais efetivamente enviados, URLs, rótulos de referência, explicações e tempos individuais estão em [examples.json](../reports/audit_2026-10-08/examples.json) e [examples.csv](../reports/audit_2026-10-08/examples.csv).

Confiança da classe prevista = `max(prob_fake, 1-prob_fake)`. É o score do modelo para sua própria decisão, sem garantia de calibração externa. `taxa_confiabilidade` da API é somente `100×(1-prob_fake)`; não é confiança da classe prevista quando a decisão é “falsa”. O score de viés pertence a outra tarefa.

| ID | Texto/variação avaliada | Referência | Classe prevista | P(fake) | Confiança da classe | Viés / score |
| --- | --- | --- | --- | ---: | ---: | --- |
| R1 | Inclusão de 82 empregadores no cadastro, em outubro de 2026 | Verdadeira | Verdadeira | 0,19% | 99,81% | opinativo / 61,00% |
| R2 | Mesmo texto, trocando 82 por 8200 | Falsa, alteração sintética | Verdadeira | 0,19% | 99,81% | opinativo / 61,56% |
| R3 | Aviso “faltou” não confirma ausência na votação | Verdadeira | Verdadeira | 9,38% | 90,62% | opinativo / 79,57% |
| R4 | Mesmo texto, removendo “não” | Falsa, alteração sintética | Verdadeira | 9,38% | 90,62% | opinativo / 83,89% |
| R5 | Interrupção na consulta de resultados não alterou a contagem | Verdadeira | Verdadeira | 27,37% | 72,63% | neutro / 66,89% |
| R6 | A interrupção modificou a contagem e o resultado presidencial | Falsa, alegação desmentida | Falsa | 56,25% | 56,25% | opinativo / 79,94% |
| E7 | “Ministério atualiza cadastro. Novos nomes foram incluídos.” | Informação insuficiente | Verdadeira | 0,08% | 99,92% | opinativo / 71,86% |
| E8 | Chamada caça-cliques preservando o fato de R1 | Verdadeira, redação sintética | Verdadeira | 0,04% | 99,96% | opinativo / 82,63% |
| E9 | Ironia sobre urnas com “bola de cristal” | Sarcasmo, fora do rótulo binário | Verdadeira | 23,25% | 76,75% | opinativo / 98,68% |
| E10 | Equador preparava cova coletiva em março de 2020 | Verdadeira no contexto histórico, segundo corpus | Verdadeira | 0,14% | 99,86% | neutro / 73,26% |

Fontes de referência: [publicação do MTE via Agência Gov, 06/10](https://agenciagov.ebc.com.br/noticias/202610/201clista-suja201d-do-trabalho-analogo-a-escravidao-e-atualizada-com-82-novos-empregadores), [esclarecimento do TSE sobre “faltou”, 07/10](https://www.tse.jus.br/comunicacao/radio/2026/Outubro/fato-ou-boato-mensagem-faltou-nao-significa-que-eleitor-deixou-de-votar) e [esclarecimento do TSE sobre a apuração, 06/10](https://www.tse.jus.br/comunicacao/radio/2026/Outubro/fato-ou-boato-justica-eleitoral-totaliza-e-divulga-os-resultados-no-mesmo-dia-da-eleicao). Para E10, a referência é o rótulo e a data do dataset local, associado à URL histórica do UOL; não foi feita nova confirmação independente desse artigo.

Nos oito casos com referência binária, a matriz exploratória é `[[1,2],[0,5]]`, com seis acertos (75%). Os dois casos sem referência binária foram excluídos. Esse percentual tem amostra mínima, exemplos relacionados e alterações construídas; **não estima a acurácia em produção**.

As falhas observadas têm causas concretas:

- **Números e negação:** R1/R2 e R3/R4 tornam-se pares de strings idênticas para o SVC. O tom muda no Transformer, mas ele não corrige `prob_fake`. Preservar essas informações é requisito para o próximo treino.
- **Fragmentos:** E7 recebe confiança de 99,92% mesmo sem conteúdo suficiente para confirmar uma notícia. Não existe classe “inconclusivo”, validação de quantidade mínima de conteúdo ou rejeição de texto fora do domínio.
- **Clickbait:** E8 recebe viés mais alto e score factual favorável. O sistema separa tom e factualidade, mas não detecta necessariamente uma chamada enganosa ou contradição entre título e corpo.
- **Sarcasmo:** E9 é classificado como verdadeiro; o rótulo binário não representa essa tarefa. Não há detector validado de sátira/ironia.
- **Temporalidade:** E10 é classificado como verdadeiro, mas não há mecanismo que detecte a reutilização da notícia como se fosse atual. Datas numéricas são eliminadas pelo SVC e o cache não expira.

### Confiança e explicabilidade

No holdout de títulos, Brier loss fake = 0,0760; log-loss = 0,2832; ECE descritivo em dez bins = 0,0966. Essas medidas são dependentes da amostra auditada e não certificam calibração em notícias recentes. A calibração sigmoid sozinha não resolve mudança de domínio. Os bins e métricas completas estão em [metrics.json](../reports/audit_2026-10-08/metrics.json).

A explicação em [ml/xai.py](../ml/xai.py) assume que coeficiente negativo aponta para fake. Neste SVC, coeficiente positivo aumenta a margem da classe codificada como fake. O calibrador tem `a=4,456838` e `b=-2,595052`: sua reorientação faz a probabilidade de fake aumentar com a margem, confirmando a direção. Por isso os textos explicativos de “sustenta”/“enfraquece” estão invertidos para este artefato. O calibrador funciona; a interpretação dos coeficientes não acompanha sua semântica.

Também foi reproduzido que `https://gov.br.evil.example/` recebe a descrição de fonte oficial: a regra usa substring no hostname. O campo “corroboração” descreve apenas o tom da escrita, sem consultar outra fonte. Essas justificativas não devem ser apresentadas como confirmação de fatos.

### Teste cego com notícias da última semana

Não foi encontrado registro de teste cego anterior com notícias de **01–07/10/2026**. Nesta investigação foi realizado o piloto recente descrito acima, com fontes datadas, rótulos fixados antes das previsões e sem treinamento. Ele revela falhas, mas não substitui um teste cego independente: os textos foram resumidos/construídos pelo investigador, incluem pares relacionados e não representam artigos íntegros ou a distribuição do produto.

Requisito para validar generalização: congelar modelo e threshold; formar um corpus novo com título, corpo, data, fonte, alegação e evidência; remover duplicatas exatas/aproximadas e parentes de artigos do treino; separar por tempo e agrupar por narrativa/fonte. A anotação deve ocorrer sem acesso à predição, com dois revisores e arbitragem de discordâncias. Começar com pelo menos 1.000 exemplos, idealmente 500 falsas e 500 verdadeiras para análise por classe, incluindo saúde, política, outras áreas e edge cases. Publicar intervalos de confiança e avaliar também a prevalência real de uso; um corpus balanceado não basta para prever a precisão operacional.

## 3. Planejamento de deploy e produção

### Infraestrutura, latência e custo

A execução real usou CPU em macOS ARM64, Python 3.14.6, scikit-learn 1.9.1, Torch 2.14.1 e Transformers 5.19.0. O SVC foi serializado com scikit-learn 1.9.0 e emitiu `InconsistentVersionWarning`. Fixar ambiente compatível com o treino faz parte da liberação. [Scikit-learn: compatibilidade de persistência](https://scikit-learn.org/stable/model_persistence.html).

| Medida | Resultado local | O que foi incluído |
| --- | ---: | --- |
| Inferência linear aquecida, p50 / p95 | 0,64 / 0,83 ms | Limpeza + `predict_proba`; 30 chamadas sequenciais |
| Inferência híbrida aquecida, p50 / p95 | 391 / 2.260 ms | Limpeza, Transformer, SVC e XAI; 30 chamadas sequenciais |
| Maior tempo híbrido observado | 2.304 ms | Carga de texto misto, incluindo textos longos |
| Carregamento com cache de pesos pronto | 0,66 s | Dentro de `load_model`; exclui importação das bibliotecas e download |
| Parâmetros do Transformer | 278.811.651 | Contagem do modelo efetivamente carregado |
| Bytes dos parâmetros carregados | 557.623.302 | Cerca de 0,56 GB; não inclui ativações e bibliotecas |
| Pico RSS do processo de auditoria | 1.326.727.168 bytes | Cerca de 1,24 GiB; inclui dados e ferramentas de avaliação |

O benchmark alternou os dez exemplos curtos com texto repetido longo; o Transformer manteve o corte de 1.500 caracteres. Não houve concorrência. Rede, scraping, acesso ao banco, cache e o tempo total HTTP ficaram fora dessa medição. O scraper tem timeout de 10 s por operação HTTP; isso não é um SLA total da requisição. Não foi medido cold start de uma imagem limpa nem benchmark numa máquina cloud.

**GPU dedicada não é requisito para executar este protótipo:** a inferência com ambos os modelos foi demonstrada em CPU. A necessidade de GPU só pode ser decidida depois de definir volume, concorrência e SLA e medir no ambiente de destino. Um serviço de inferência CPU com limite de concorrência é suficiente para começar os testes de homologação.

Como configuração inicial de homologação, reservar 2 vCPUs e 4–8 GB de RAM para um processo híbrido, com PostgreSQL separado e disco suficiente para imagem, cache e logs. Trata-se de hipótese de dimensionamento, não capacidade garantida. Cada processo com modelos próprios aumenta memória. Evitar escalar o número de workers sem medir RAM e competição por CPU. Versões de modelo/Hugging Face devem ser fixadas; a imagem deve conter os artefatos para permitir startup sem internet. O Dockerfile baixa pesos no build, mas ainda carrega os tensores no startup.

Não há requisito de “tempo real” definido no projeto. Para uma meta hipotética de p95 de inferência abaixo de 2 s, o híbrido não passou neste ensaio. Para experiência interativa com resposta total abaixo de 5 s, o resultado isolado é promissor, mas faltam scraping, rede, banco e concorrência para comprovar atendimento. Separar viés em execução opcional/assíncrona pode reduzir custo e espera, se o contrato de produto aceitar essa mudança; isso exige validação, não foi implementado.

Referência de orçamento, consultada em 08/10/2026: a tabela pública do Lightsail anuncia Linux com IPv4, 2 vCPUs/4 GB por US$24/mês e 2 vCPUs/8 GB por US$44/mês; banco gerenciado de 2 GB com criptografia por US$30/mês ou US$60 em alta disponibilidade; balanceador por US$18/mês. [Preços oficiais AWS Lightsail](https://aws.amazon.com/lightsail/pricing/).

| Cenário ilustrativo | Composição | Subtotal mensal |
| --- | --- | ---: |
| Homologação com folga de RAM | Uma VM de 8 GB + banco de 2 GB padrão | US$74 |
| Piloto com redundância de aplicação e banco | Duas VMs de 8 GB + banco de 2 GB em alta disponibilidade + balanceador | US$166 |

Os subtotais são aritmética sobre os preços publicados, não proposta comercial ou aprovação de infraestrutura. Excluem snapshots, excedentes, impostos, câmbio, observabilidade externa, trabalho de anotação e retreino. Confirmar disponibilidade e preço da configuração na região escolhida; a página informa franquia de transferência reduzida em São Paulo. Não há volume de tráfego para calcular custo por notícia. A fórmula operacional é custo fixo mensal + armazenamento/backup + excedentes + anotação humana, dividido pelo número de análises novas bem-sucedidas; cache altera esse denominador.

### Monitoramento e retreinamento

O estado observado é inferior ao descrito no roadmap:

| Componente | Evidência atual | Lacuna para produção |
| --- | --- | --- |
| DVC | `data.dvc` e diretório `.dvc` | `.dvc/config` sem remote; recuperação remota não demonstrada |
| MLflow | Código de registro e banco local | Banco inspecionado sem runs/modelos; treino aponta para outro arquivo; API não usa registry |
| Evidently | `ml/drift_monitor.py` | Import falha: `No module named 'evidently.report'`, com Evidently 0.7.23 |
| Referência de drift | `Classe` histórica comparada a `score_ml` atual | Rótulo de verdade e score de fake são variáveis diferentes, com polaridade incompatível |
| Janelas/alertas | Consulta de todo o cache local | Sem janela temporal, tamanho mínimo, alerta integrado ou agendamento no repositório |
| Feedback | `/votar`, tabela de votos | Zero votos; falta validação de domínio dos valores, revisão e proteção contra votos repetidos |
| Retreino | `ml/retrain_cron.py` | Usa SQLite fixo mesmo no Compose PostgreSQL; incorpora votos diretamente; não promove modelo servido |

O script de retreino lê todos os votos, não apenas discordâncias revisadas. O valor `-1`, aceito pelo contrato como “discorda”, pode virar uma terceira classe se entrar no dataset. Vários votos conflitantes para o mesmo texto acabam resolvidos pelo último registro após deduplicação, sem agregação de reputação ou arbitragem humana. Não há pipeline de decisão “challenger supera champion → promover”; registrar um modelo no MLflow não altera o `.joblib` da API.

Data drift pode sinalizar mudança na distribuição de textos ou scores; concept drift exige verificar mudança na relação entre informação e rótulo, com ground truth recente. Um detector de distribuição sozinho não prova queda de acurácia. [Evidently: explicação de data drift](https://docs.evidentlyai.com/metrics/explainer_drift).

Estratégia proposta:

1. Registrar versão do modelo/dataset/limpeza, hash do conteúdo, timestamps de publicação e análise, domínio, comprimento, cobertura do vocabulário, `prob_fake`, modo normal/fallback e latências por etapa. Versionar e expirar cache por URL + hash de conteúdo + modelo.
2. Monitorar diariamente erro de extração, vazios, fallback, latência p50/p95/p99, fila, CPU/RAM e distribuição de fontes/scores. Comparar features e scores calculados pelo mesmo modelo nos dados de referência e atuais, usando janelas e amostra mínima documentadas.
3. Revisar semanalmente amostras aleatórias e casos incertos, com rótulos humanos baseados em fontes. Calcular precisão, recall, FPR/FNR e calibração por período, tema e origem. O desempenho deve incluir a taxa de abstenção e os casos que não puderam ser verificados.
4. Criar fila de revisão dos votos; validar valores, agregação, identidades/reputação e referências antes de incorporar. Feedback sem revisão é sinal de investigação, não ground truth para retreino.
5. Executar retreino mensal ou quando houver degradação confirmada, com dados versionados, holdout temporal congelado e conjuntos separados para calibração e seleção de threshold. Comparar champion/challenger em todas as metas, publicar uma versão imutável e fazer canário com rollback.

Gatilhos iniciais propostos: investigar aumento de OOV de 10 pontos percentuais ou mudança persistente de scores/fontes; abrir revisão se recall anotado cair 5 pontos ou sair da meta de liberação. São valores para ajuste no piloto, não regras já ativas ou prova estatística de drift. Nunca promover automaticamente apenas por um teste de distribuição.

### Bloqueios, critérios de liberação e cronograma

Prioridades P0 impedem afirmar verificação confiável de fatos; P1 impedem operação estável e reproduzível.

| Prioridade | Bloqueio | Critério concreto de conclusão | Responsável sugerido |
| --- | --- | --- | --- |
| P0 | Título/corpo e rótulos inconsistentes | Schema explícito; mapping persistido; conjunto de treino com a mesma entrada da API | ML + engenharia de dados |
| P0 | Calibração/holdout contaminados | Split por artigo/narrativa e tempo; treino, calibração e teste disjuntos | ML + dados |
| P0 | Remoção de números, datas e negação | Representação preserva informação; novo treino supera avaliação adversarial | ML |
| P0 | Score apresentado como confirmação factual | Contrato de triagem com abstenção, ou etapa de checagem com evidências e temporalidade validada | Produto + ML |
| P0 | XAI invertido e regra de domínio vulnerável | Atribuição acompanha o artefato calibrado; hostname validado; texto descreve associação lexical | ML + backend |
| P0 | Falta de teste recente independente | Corpus anotado, matriz, intervalos e métricas por grupos com modelo congelado | Anotação + ML |
| P1 | Instalação/container não reproduzíveis | Lock consistente, dependências completas, build e startup offline em CI | Backend + MLOps |
| P1 | Registry não promove à API | Manifesto/model version, deploy canário e rollback testados | MLOps |
| P1 | Drift/retreino/feedback incompletos | Monitor executável; janelas/alertas; votos revisados; agenda e promoção verificáveis | MLOps + backend |
| P1 | URL arbitrária, cache eterno e readiness incompleta | Bloquear rede privada/local e redirects indevidos; limites; TTL/model version; readiness dos dois modelos | Backend |
| P1 | Falta de teste sob carga e recuperação | SLA medido com scraping/cache/concorrência; falhas e restore de banco exercitados | Backend + infraestrutura |

Os bloqueios de dependência são específicos: `nltk` está no `pyproject.toml`, mas ausente do lock inspecionado; pandas, MLflow, DVC e Evidently não estão declarados no projeto/lock para seus scripts. A venv tem NumPy 2.5.3, enquanto o projeto declara `^1.26.0` e o lock registra 1.26.4. Poetry não está disponível no PATH para executar `poetry check`; o mismatch de nltk foi identificado por leitura direta dos manifests. Atualizar a compatibilidade deve preceder o build limpo. Downloads NLTK no import e alteração global do contexto TLS em caso de falta de stopwords também precisam sair do caminho de runtime.

Critérios de qualidade **propostos**, a serem ratificados por produto conforme o custo de cada erro: precisão fake ≥95%, recall fake ≥90%, FPR ≤2% e ECE ≤0,05 em teste recente independente, com intervalos e comportamento por tema publicados. Não basta atingir essas metas na média se um grupo relevante falhar. Uma meta pode exigir threshold diferente ou mais abstenção; devem ser escolhidos em validação, antes de abrir o teste final. O modelo atual não tem evidência de cumprir esses critérios.

Estimativa a partir da disponibilização de revisores e ambiente de homologação:

| Período | Entrega | Condição para avançar |
| --- | --- | --- |
| Semana 1 | Contratos de entrada/rótulos, correção de dados e pipeline reproduzível | Mesma entrada no treino/serving; splits e hashes auditáveis |
| Semanas 1–2, em paralelo | Corpus recente e guia de anotação | Revisão independente e adjudicação concluídas |
| Semana 2–3 | Treino/calibração disjuntos, XAI e threshold | Metas em validação e edge cases; teste final preservado |
| Semana 3–4 | Container/CI, PostgreSQL, registry e observabilidade | Startup, migrações, fallback e rollback comprovados |
| Semana 4–5 | Homologação com carga e teste cego final | Metas de qualidade e SLA atendidos com intervalos publicados |
| Semana 5–6 | Piloto restrito e possível v1.0 | Monitoramento, revisão e rollback operacionais |

O intervalo de **4–6 semanas** é uma estimativa de engenharia, não compromisso de lançamento. Falta de rótulos/evidências ou necessidade de uma camada completa de checagem factual pode ampliar o prazo. A v1.0 deve ser liberada por critérios atendidos, não apenas por data.

## Entregáveis e reprodução

- [Métricas completas e hashes](../reports/audit_2026-10-08/metrics.json).
- [Matriz de confusão CSV](../reports/audit_2026-10-08/confusion_matrix.csv) e [figura SVG exportável](../reports/audit_2026-10-08/confusion_matrix.svg).
- [Dez exemplos CSV](../reports/audit_2026-10-08/examples.csv) e [JSON com textos, scores e XAI](../reports/audit_2026-10-08/examples.json).
- [Todas as previsões do holdout](../reports/audit_2026-10-08/title_holdout_predictions.csv) e [do corpus de calibração](../reports/audit_2026-10-08/full_calibration_corpus_predictions.csv).
- [Script de auditoria](../scripts/audit_model.py), sem treino ou alteração do serviço.

Reproduzir a partir da raiz, com a venv existente e os pesos/dados já presentes:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/audit_model.py
```

O script sobrescreve somente os artefatos de diagnóstico em `reports/audit_2026-10-08/`. O benchmark varia por máquina/carga. Fontes externas foram consultadas para definir os casos, mas não são baixadas pelo script. Os arquivos contêm as previsões efetivas, não scores estimados ou simulados.
