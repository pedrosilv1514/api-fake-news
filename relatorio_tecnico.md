# Relatório Técnico — Pipeline de Classificação de Fake News

**Programa:** Residência em Inteligência Artificial — Universidade de Brasília (UnB)  
**Projeto:** Identificador de Fake News / Hub de Verificação de Notícias  
**Data de referência:** 9 de outubro de 2026  
**Escopo:** preparação de dados, modelagem, avaliação, explicabilidade e operação de modelos de processamento de linguagem natural (NLP).

## Resumo

O projeto investiga a classificação de notícias em português por meio de uma arquitetura com duas famílias de modelos: uma baseline de TF-IDF e LinearSVC calibrado, de menor custo computacional e interpretação direta, e transformers multilingues, representados pelo mDeBERTa-v3. O objetivo é apoiar a triagem de conteúdos potencialmente desinformativos e a revisão humana, com rastreabilidade dos dados e dos experimentos.

Este relatório distingue implementação observável, evolução proposta e resultados hipotéticos. O repositório contém treinamento de candidatos lineares, versionamento inicial com DVC, API FastAPI e análise opcional de tom com mDeBERTa em modo zero-shot. Fine-tuning supervisionado para classificação de Fake News, integração do treinamento com MLflow e implantação em AWS/GCP são apresentados como evolução arquitetural. A comparação quantitativa entre as famílias utiliza números hipotéticos, explicitamente identificados. Resultados históricos existentes aparecem separadamente.

A principal restrição científica é que padrões linguísticos associados ao rótulo FAKE não comprovam a falsidade de uma alegação. A solução deve comunicar essa diferença e complementar a classificação textual com evidências externas e revisão especializada.

## 1. Contexto, objetivos e estado do desenvolvimento

O problema foi formulado como classificação supervisionada binária, com o contrato de rótulos **0 = FAKE e 1 = REAL**. Nas métricas deste documento, FAKE é a classe positiva, independentemente de seu valor numérico. Essa definição deve acompanhar treinamento, calibração, avaliação, explicações e resposta da API.

Os objetivos técnicos são estabelecer uma baseline reproduzível, investigar ganhos de representação contextual, tornar predições auditáveis e organizar o ciclo de vida dos modelos. A qualidade precisa ser examinada junto com custo, latência, cobertura e consequências dos erros.

| Componente | Situação observada | Evolução discutida |
| --- | --- | --- |
| Dados e DVC | Diretório data/ referenciado por data.dvc; configuração remota consultada sem remoto definido | Armazenamento remoto, pipeline declarativo e vínculo entre dados e experimentos |
| Modelo linear | TF-IDF, LinearSVC e calibração; candidatos separados do modelo servido | Seleção de hiperparâmetros e validação independente recente |
| Transformer | mDeBERTa NLI opcional, zero-shot, para análise auxiliar de tom | Fine-tuning binário e comparação controlada com a baseline |
| MLflow | Banco local com experimento Default, sem runs, métricas ou modelos registrados | Tracking integrado e governança no Model Registry |
| API | FastAPI, endpoints legados e v2, explicações e protótipo de verificação por evidências | Homologação, observabilidade e implantação em nuvem |
| Retreinamento | Geração de candidatos; retreino automático com votos não revisados desativado | Incorporação de dados revisados e promoção controlada |

Essas constatações se baseiam no código e nos artefatos locais, especialmente [README](README.md), [treinamento](ml/train.py), [inferência](ml/inference.py) e [entrega técnica](docs/ENTREGA_EVOLUCAO_2026-10-08.md). Documentos anteriores contêm afirmações mais amplas de prontidão; elas não substituem evidências de execução, avaliação e implantação.

## 2. Preparação, coleta e versionamento dos dados

### 2.1. Origem e coleta

O acervo local identificado como FakeRecogna contém 11.902 registros, originalmente distribuídos em 5.951 exemplos FAKE e 5.951 REAL, conforme a [investigação do corpus](docs/INVESTIGACAO_MODELO_2026-10-08.md). O arquivo fake_recogna_limpo.csv fornece os campos utilizados no treinamento: título, notícia, classe, data e URL. A presença do arquivo não comprova sua procedência completa; a documentação de origem deve acompanhar cada versão.

Na análise de URLs, a aplicação utiliza httpx e trafilatura para obter o conteúdo textual e remover elementos de navegação. Esse fluxo atende à inferência e à consulta de evidências. Sua saída não deve ingressar automaticamente no conjunto supervisionado: cada incorporação exige verificação da origem, anotação e revisão do rótulo.

Para ampliar o corpus, o processo proposto registra:

- URL, domínio, título e corpo original, além da versão extraída e normalizada;
- datas de publicação, atualização, coleta e referência factual;
- identificador do artigo, hash do conteúdo, idioma e grupo de duplicação ou narrativa;
- rótulo, evidências, responsável pela anotação e resolução de divergências;
- condições de uso da fonte e limitações da coleta.

A unidade de anotação deve ser definida previamente. Uma notícia pode combinar alegações corretas e incorretas; nesse caso, é necessário explicitar a regra de agregação para o rótulo do artigo ou manter a avaliação por alegação.

### 2.2. Limpeza e preservação da informação

O pré-processamento atual para novos candidatos, identificado como **facts-v2**, normaliza Unicode, converte texto para minúsculas, substitui URLs por um marcador e padroniza espaços. Preserva negações, números, datas, percentuais e unidades, informações essenciais para distinguir afirmações factuais. Título e corpo seguem uma composição compartilhada entre treino e inferência candidata. A implementação está em [ml/preprocessing.py](ml/preprocessing.py).

Essa política corrige uma limitação da limpeza legada, que elimina caracteres e stopwords de forma mais agressiva. Remover “não”, valores ou datas pode alterar o significado da notícia. O artefato legado conserva seu pré-processamento por compatibilidade com os pesos; novos modelos devem usar uma representação versionada e consistente.

Existe também uma limitação de origem: a coluna Noticia do corpus local já passou por transformação. A normalização atual preserva o conteúdo restante, mas não recupera informações eliminadas anteriormente. Uma evolução confiável exige obter e versionar os textos originais.

O truncamento de 1.500 caracteres encontrado na inferência refere-se ao módulo opcional de tom. Não constitui uma regra geral de limpeza nem corresponde a 1.500 tokens. No transformer supervisionado, propõe-se controlar a extensão pelo tokenizer e avaliar a perda de trechos do documento.

### 2.3. Deduplicação e divisão dos dados

O módulo [ml/dataset.py](ml/dataset.py) valida campos obrigatórios e rótulos, exclui registros sem data interpretável ou corpo e identifica duplicatas e conflitos. O agrupamento considera corpos ou URLs iguais e títulos aproximadamente semelhantes, com Jaccard de pelo menos 0,85 sobre candidatos selecionados por termos raros.

Os grupos são separados cronologicamente em treino, calibração e teste. Grupos que atravessam fronteiras temporais são expurgados para reduzir vazamento. Os cortes derivam das datas dos grupos; as proporções finais de artigos não equivalem necessariamente a 60%/20%/20%.

Para o candidato histórico examinado, o manifesto registra:

| Partição | Artigos | FAKE | REAL | Período |
| --- | ---: | ---: | ---: | --- |
| Treino | 5.159 | 2.862 | 2.297 | 09/02/2012 a 22/07/2020 |
| Calibração | 3.530 | 1.373 | 2.157 | 23/07/2020 a 08/02/2021 |
| Teste histórico | 2.242 | 968 | 1.274 | 09/02/2021 a 07/09/2021 |

Foram excluídos 963 registros sem data interpretável e oito registros de grupos nas fronteiras. A heurística reduz contaminação, mas não comprova independência entre todas as narrativas. Recomenda-se complementar o protocolo com revisão semântica e avaliação por fontes não vistas.

Para selecionar hiperparâmetros, deve haver validação temporal interna ao treino. A calibração permanece independente, e um novo teste recente deve ficar reservado até o encerramento do desenvolvimento. O teste histórico já observado não pode ser tratado como avaliação cega final.

### 2.4. Versionamento com DVC

O arquivo data.dvc registra metadados e hash do diretório de dados. O Git acompanha esse ponteiro; os conteúdos são mantidos no cache do DVC e, quando configurado, no armazenamento remoto. Esse primeiro mecanismo existe, mas não foram encontrados dvc.yaml ou dvc.lock na inspeção.

O treinamento atual utiliza, por padrão, o CSV na raiz do projeto, enquanto o ponteiro DVC cobre data/. Essa diferença exige alinhar o caminho de entrada ao conjunto versionado ou comprovar a correspondência por hash; a existência do ponteiro não garante que toda execução utilize os dados rastreados.

A evolução proposta utiliza estágios de preparação, divisão, treinamento e avaliação. Dependências, parâmetros e saídas devem ser declarados para que dvc repro reexecute os estágios afetados por mudanças. A definição segue o modelo de [pipelines do DVC](https://doc.dvc.org/user-guide/pipelines/defining-pipelines).

| Estágio proposto | Dependências principais | Saídas versionadas |
| --- | --- | --- |
| Preparar | Corpus bruto, regras de anotação e código de limpeza | Corpus validado e relatório de qualidade |
| Dividir | Corpus validado, agrupamento e cortes temporais | IDs e metadados das partições |
| Treinar linear | Treino, calibração e parâmetros TF-IDF/SVC | Pipeline calibrado e manifesto |
| Ajustar transformer | Treino, validação e revisão do checkpoint | Pesos, tokenizer e configuração |
| Avaliar | Modelos congelados e teste reservado | Métricas, matrizes e predições por artigo |

S3 ou Google Cloud Storage são opções de remoto suportadas pelo DVC. Credenciais devem ficar fora do Git, e o upload precisa ser acompanhado de verificação de recuperação em ambiente limpo. Cada versão deve vincular commit, hash dos dados, configuração e identificador do experimento. Consulte a documentação de [armazenamento remoto do DVC](https://doc.dvc.org/user-guide/data-management/remote-storage).

## 3. Pipeline de treinamento e configuração dos modelos

### 3.1. Fluxo geral

~~~mermaid
flowchart TD
    A[Corpus e anotação revisada] --> B[Validação, normalização e agrupamento]
    B --> C[Partições temporais versionadas]
    C --> D[TF-IDF + LinearSVC + calibração]
    C --> E[Fine-tuning mDeBERTa-v3: proposto]
    D --> F[Avaliação comum e análise de erros]
    E --> F
    F --> G[Tracking e registro MLflow: propostos]
    G --> H[Candidato aprovado e artefato imutável]
    H --> I[FastAPI e infraestrutura AWS ou GCP]
    I --> J[Monitoramento e revisão humana]
    J --> A
~~~

O diagrama representa a arquitetura de referência completa. A comparação exige os mesmos artigos de teste, contrato de rótulos e política de decisão. O treinamento não deve sobrescrever o modelo servido; o fluxo linear atual já gera candidatos em diretórios próprios, com manifesto, hashes, partições e predições.

### 3.2. Baseline: TF-IDF e LinearSVC

TF-IDF representa textos em vetores esparsos, ponderando termos pela frequência no documento e frequência inversa no corpus. A baseline utiliza unigramas e bigramas, até 50.000 atributos e exclusão de termos presentes em mais de 95% dos documentos. Essa parametrização corresponde às opções de [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

O trecho reproduz o núcleo do treinamento candidato atual. As variáveis representam partições previamente preparadas; o exemplo não realiza coleta ou divisão.

~~~python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.preprocessing import normalizar_texto

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=normalizar_texto,
        lowercase=False,
        token_pattern=r"(?u)R\$|[€$£]|\b\w+(?:[.,/-]\w+)*(?:%|‰|°[cCfF])?",
        ngram_range=(1, 2),
        max_features=50_000,
        max_df=0.95,
    )),
    ("clf", LinearSVC(C=1.0, max_iter=5_000, random_state=42)),
])

pipeline.fit(X_train, y_train)
modelo = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
modelo.fit(X_calibration, y_calibration)

indice_fake = list(modelo.classes_).index(0)
prob_fake = modelo.predict_proba(X_test)[:, indice_fake]
~~~

Vocabulário e pesos IDF são aprendidos somente no treino. O LinearSVC produz uma margem de decisão; a calibração sigmoid ajusta sua conversão para probabilidades em outra partição. Aplicar uma sigmoid arbitrária à margem não substitui esse ajuste. A independência dos dados de calibração é uma exigência metodológica descrita pelo [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).

Como evolução, recomenda-se selecionar C, extensão dos n-gramas e limites do vocabulário em validação temporal interna. A configuração atual utiliza C=1 e não ativa pesos de classe balanceados; seu uso deve ser comparado experimentalmente.

### 3.3. Transformers multilingues: zero-shot e fine-tuning

O mDeBERTa-v3 oferece representações contextuais multilingues que motivam sua investigação para relações semânticas menos dependentes da ocorrência exata de palavras. A família DeBERTa-v3 utiliza pré-treinamento com detecção de tokens substituídos, conforme o [artigo original](https://arxiv.org/abs/2111.09543). Essa propriedade fundamenta uma hipótese de ganho; não demonstra superioridade no corpus do projeto.

O checkpoint atualmente referenciado, MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, foi ajustado para inferência de linguagem natural (NLI) e permite classificação zero-shot pela comparação entre texto e hipóteses. No projeto, classifica tom neutro ou opinativo/sensacionalista. Essa tarefa difere da classificação supervisionada FAKE/REAL. A avaliação publicada no [model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) não comprova desempenho em notícias brasileiras.

Para o fine-tuning proposto, o protocolo é inicializar o encoder multilingue e uma cabeça binária, treinar com artigos anotados e avaliar em português. Uma opção é partir de microsoft/mdeberta-v3-base; outra é comparar a inicialização pelo checkpoint NLI, substituindo sua cabeça de três classes. Origem e revisão dos pesos devem ser fixadas antes da execução.

Configuração inicial de referência, ainda não executada como experimento comparativo neste repositório:

~~~yaml
transformer:
  checkpoint: microsoft/mdeberta-v3-base
  revision: <commit-do-checkpoint-a-fixar>
  num_labels: 2
  label_mapping: {0: FAKE, 1: REAL}
  max_length: 512
  padding: dynamic
  learning_rate: 2.0e-5
  epochs_max: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  optimizer: AdamW
  weight_decay: 0.01
  warmup_ratio: 0.10
  early_stopping_patience: 2
  selection_metric: macro_f1
  evaluation_strategy: epoch
  seed: 42
  precision_initial: fp32
~~~

O bloco define intenções experimentais; não é um arquivo diretamente consumido pelo Trainer. A acumulação corresponde a 32 artigos por passo de atualização com um dispositivo. A implementação deve mapear os campos para a versão fixada da biblioteca, tokenizar título e corpo, calcular a perda e selecionar o checkpoint somente pela validação. O fluxo de ajuste supervisionado está documentado pela [Hugging Face](https://huggingface.co/docs/transformers/tasks/sequence_classification).

Para artigos longos, devem ser comparados truncamento e janelas sobrepostas, registrando a agregação das predições por artigo. Todas as janelas de uma notícia pertencem à mesma partição. Precisão reduzida é uma otimização posterior, condicionada à estabilidade numérica do checkpoint e do hardware. Calibração e escolha dos limiares também precisam usar dados separados da seleção do checkpoint e do teste final.

## 4. Critérios de seleção da abordagem

### 4.1. Complementaridade e decisão arquitetural

| Critério | TF-IDF + LinearSVC | mDeBERTa-v3 supervisionado |
| --- | --- | --- |
| Representação | Padrões lexicais e n-gramas | Representação contextual multilingue |
| Treinamento | Menor custo; execução viável em CPU | Maior demanda computacional e de ajuste |
| Inferência | Operação esparsa adequada à baseline online | Custo dependente de tokens, hardware e concorrência |
| Explicabilidade | Decomposição direta da margem por atributo | Atribuição aproximada e análise de perturbações |
| Generalização | Sensível a mudanças de vocabulário e domínio | Potencial de transferência; requer validação local |
| Operação | Artefato e dependências relativamente simples | Pesos, tokenizer, runtime e recursos mais exigentes |

A arquitetura híbrida designa a coexistência coordenada de componentes. No estado atual, o modelo linear e o transformer de tom têm funções distintas; não há evidência de um ensemble que combine probabilidades FAKE/REAL de ambos.

Após o fine-tuning, pode-se investigar uma cascata: a baseline realiza a triagem inicial, e casos de maior incerteza seguem para o transformer. Divergências e ausência de evidências seguem para revisão. Pesos de combinação, encaminhamento e limiares precisam ser selecionados em validação e avaliados como um sistema completo. Uma explicação do linear não explica automaticamente a decisão do transformer.

A escolha final depende de ganho estatístico e operacional: desempenho por classe, estabilidade em novas fontes, custo por requisição, latência e possibilidade de auditoria. O transformer deve ser promovido quando o ganho justificar a complexidade adicional.

### 4.2. Explainable AI — XAI

Para o modelo linear, a margem pode ser decomposta como **f(x) = b + Σ(w_j × x_j)**. Cada produto expressa a contribuição de um termo ou n-grama. O motor [ml/xai.py](ml/xai.py) extrai essas contribuições e considera a orientação do calibrador para apresentá-las na direção de FAKE.

As contribuições explicam o comportamento matemático do classificador e ajudam a localizar vieses ou dependência excessiva de termos. Não são causas da falsidade nem parcelas aditivas da probabilidade calibrada. Uma palavra frequente em boatos pode aparecer em reportagem legítima que os desmente.

Para o transformer, propõe-se investigar Integrated Gradients sobre embeddings, com baseline explicitada, e verificar estabilidade por perturbações controladas. O método está disponível no [Captum](https://captum.ai/docs/extension/integrated_gradients). As atribuições devem ser avaliadas quanto à fidelidade; mapas de atenção isolados não bastam para justificar uma decisão.

A resposta ao usuário deve distinguir classificação textual, análise de tom, trechos de evidências e limitações. Versão do modelo e pré-processamento precisam acompanhar a trilha de auditoria. A explicação deve corresponder ao componente que efetivamente determinou o resultado.

### 4.3. Infraestrutura AWS/GCP e endpoints FastAPI

A opção por nuvem decorre da necessidade de armazenar artefatos, executar treinamentos com recursos adequados e escalar a API independentemente dos experimentos. AWS e GCP constituem alternativas e permitem explorar portabilidade; utilizar ambas simultaneamente exige justificativa de custo ou capacidade.

| Camada | Opção AWS | Opção GCP | Critério técnico |
| --- | --- | --- | --- |
| Dados e artefatos | Amazon S3 | Google Cloud Storage | Objetos versionados, acesso controlado e integração DVC |
| API conteinerizada | Amazon ECS/Fargate para a camada CPU | Cloud Run | Escala do serviço e isolamento de dependências |
| Transformer acelerado | Serviço ou instância com GPU, separado da API quando necessário | Cloud Run com GPU, quando compatível | Memória, latência e disponibilidade regional |

As opções de objetos são descritas pela [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) e pelo [Google Cloud](https://docs.cloud.google.com/storage/docs/introduction). A documentação de [Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html) e de [GPU no Cloud Run](https://docs.cloud.google.com/run/docs/configuring/services/gpu) deve orientar a escolha efetiva. A tabela é uma proposta; não registra deploy realizado, custos contratados ou desempenho homologado.

FastAPI fornece contratos de entrada e saída e integra os componentes existentes:

- **POST /analisar-url:** análise textual legada e campos compatíveis com o frontend;
- **POST /v2/analisar-url:** análise de URL com resultado de verificação separado;
- **POST /v2/verificar:** fluxo detalhado de alegações e evidências;
- **GET /health e GET /ready:** saúde do serviço e prontidão do classificador legado.

O carregamento via lifespan permite inicializar recursos antes de atender requisições, conforme a [documentação do FastAPI](https://fastapi.tiangolo.com/advanced/events/). Na implementação, ocorre por processo; múltiplos workers podem multiplicar a memória dos pesos. Inferência custosa exige concorrência limitada e capacidade dimensionada. O uso de async não acelera o cálculo neural.

O Dockerfile atual instala o grupo principal de dependências e desativa o modelo de tom. Uma imagem com transformer exige provisionamento adicional de pesos e bibliotecas. A arquitetura proposta fixa versões e hashes na implantação, evitando downloads no startup e troca silenciosa de modelo em execução.

## 5. Rastreamento de experimentos com MLflow

### 5.1. Evidência disponível e integração proposta

A inspeção de mlflow.db encontrou somente o experimento Default: zero runs, parâmetros, métricas e modelos registrados. Também não foram encontradas chamadas de tracking no treinamento atual. Assim, não há evidência local para afirmar que os resultados existentes foram rastreados pelo MLflow.

Na evolução proposta, cada configuração recebe uma run. São registrados parâmetros, métricas, hashes dos dados, commit, seed e artefatos. A comparação hipotética da próxima seção representa o tipo de análise a produzir após a integração; seus números não foram extraídos do banco. Esse desenho segue as funções de [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).

Resultados sintéticos eventualmente registrados devem receber identificação explícita e permanecer separados dos experimentos medidos, sem elegibilidade para promoção de modelos.

| Registro proposto | Conteúdo |
| --- | --- |
| Parâmetros | Família, hiperparâmetros, tokenizer, extensão, calibração e limiares |
| Identificação | Commit Git, versão DVC, hashes de dados/splits, seed e revisão dos pesos |
| Métricas | Acurácia, precisão, recall, F1 por classe e macro, erros e cobertura |
| Artefatos | Matriz de confusão, predições por artigo, configuração e manifesto |
| Operação | Hardware, tempo de treino, memória, latência e configuração de carga |

Exemplo de instrumentação proposta, a incorporar ao treino; os argumentos devem ser produzidos pelo pipeline:

~~~python
import mlflow

def registrar_experimento(config, metricas, diretorio_artefatos):
    mlflow.set_experiment("residencia-unb-fake-news")
    with mlflow.start_run(run_name=config["run_name"]) as run:
        mlflow.log_params(config["params"])
        mlflow.set_tags(config["tags"])
        mlflow.log_metrics(metricas)
        mlflow.log_artifacts(str(diretorio_artefatos))
        return run.info.run_id
~~~

### 5.2. Registro e promoção

O Model Registry deve associar versões aprovadas às runs e aos artefatos. Um alias como champion pode indicar a referência vigente, mas a implantação deve resolver e fixar a versão concreta. Aprovação requer teste independente, critérios por classe e avaliação operacional. O modelo anterior deve permanecer disponível para rollback. Essas capacidades são descritas pelo [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/).

## 6. Métricas de desempenho e análise dos resultados

### 6.1. Definições e protocolo

Considerando FAKE como classe positiva:

- **TP:** notícia FAKE corretamente classificada como FAKE;
- **FP:** notícia REAL classificada como FAKE;
- **FN:** notícia FAKE classificada como REAL;
- **TN:** notícia REAL corretamente classificada como REAL.

| Métrica | Fórmula | Interpretação |
| --- | --- | --- |
| Acurácia | (TP + TN) / N | Fração total de acertos |
| Precisão FAKE | TP / (TP + FP) | Fração de alertas FAKE corretos |
| Recall FAKE | TP / (TP + FN) | Fração de notícias FAKE detectadas |
| F1 FAKE | 2TP / (2TP + FP + FN) | Equilíbrio entre precisão e recall |
| FPR | FP / (FP + TN) | Fração de notícias REAL indevidamente sinalizadas |
| FNR | FN / (FN + TP) | Fração de notícias FAKE não detectadas |

A avaliação real deve incluir métricas por classe, F1 macro, desempenho por tema/fonte/período e curvas de precisão-recall. Curvas de confiabilidade e Brier score complementam a avaliação probabilística; Brier isolado combina calibração e discriminação e não demonstra confiabilidade factual.

### 6.2. Comparação hipotética: linear versus transformer

**Todos os números desta subseção são hipotéticos.** Considera-se um mesmo teste de 2.000 artigos: 1.000 FAKE e 1.000 REAL. Os dois modelos tomam decisões binárias, sem abstenção, com limiar ilustrativo de 0,5 após a calibração prevista. Esse conjunto é uma construção didática e não corresponde ao teste histórico.

#### Matriz hipotética — TF-IDF + LinearSVC

Linhas representam a classe de referência; colunas, a predição.

| Referência / Predição | FAKE | REAL | Total |
| --- | ---: | ---: | ---: |
| FAKE | 900 (TP) | 100 (FN) | 1.000 |
| REAL | 80 (FP) | 920 (TN) | 1.000 |
| Total | 980 | 1.020 | 2.000 |

#### Matriz hipotética — mDeBERTa-v3 com fine-tuning

| Referência / Predição | FAKE | REAL | Total |
| --- | ---: | ---: | ---: |
| FAKE | 950 (TP) | 50 (FN) | 1.000 |
| REAL | 50 (FP) | 950 (TN) | 1.000 |
| Total | 1.000 | 1.000 | 2.000 |

#### Métricas derivadas das matrizes hipotéticas

| Métrica | TF-IDF + LinearSVC | mDeBERTa-v3 ajustado |
| --- | ---: | ---: |
| Acurácia | 91,00% | 95,00% |
| Precisão FAKE | 91,84% | 95,00% |
| Recall FAKE | 90,00% | 95,00% |
| F1 FAKE | 90,91% | 95,00% |
| FPR | 8,00% | 5,00% |
| FNR | 10,00% | 5,00% |

Nesse cenário, o transformer aumenta a acurácia em 4 pontos percentuais e o F1 FAKE em aproximadamente 4,09 pontos. Falsos negativos caem de 100 para 50, redução relativa de 50%, e falsos positivos caem de 80 para 50, redução de 37,5%. O erro total diminui de 180 para 100 artigos, redução de aproximadamente 44,4%.

Uma hipótese seria a melhor representação de contexto, paráfrases e negações pelo transformer. Os números construídos não testam essa hipótese. Em experimento real, ela exige inspeção dos erros, avaliação por recortes e controles sobre duplicação, domínio e extensão dos textos.

As matrizes agregadas não informam em quais artigos os modelos discordaram. Portanto, não permitem afirmar significância estatística. Recomenda-se avaliar predições pareadas, usar bootstrap por grupos de narrativas para intervalos de confiança e considerar McNemar quando suas premissas forem adequadas.

### 6.3. Resultados históricos efetivamente disponíveis

O candidato candidate-20261008T195757552426Z possui [métricas](models/candidates/candidate-20261008T195757552426Z/metrics.json) e [manifesto](models/candidates/candidate-20261008T195757552426Z/manifest.json) locais. Os resultados medidos são:

| Métrica histórica — modelo linear candidato | Resultado |
| --- | ---: |
| Artigos de teste | 2.242 |
| Acurácia | 94,34% |
| Precisão FAKE | 93,40% |
| Recall FAKE | 93,49% |
| F1 FAKE | 93,44% |
| F1 macro | 94,23% |
| FPR | 5,02% |
| FNR | 6,51% |
| Brier score | 0,04347 |

A matriz histórica, na mesma orientação das anteriores, é:

| Referência / Predição | FAKE | REAL |
| --- | ---: | ---: |
| FAKE | 905 | 63 |
| REAL | 64 | 1.210 |

Esses valores pertencem a outro conjunto e protocolo, não sendo comparáveis diretamente aos números hipotéticos. O manifesto registra recent_validation=false e production_approved=false. O corpus transformado e a observação do teste durante o desenvolvimento restringem a interpretação. Não há avaliação equivalente de transformer ajustado que sustente superioridade empírica neste projeto.

### 6.4. Decisão, prevalência e abstenção

O limiar deve considerar o custo dos erros. Falsos positivos podem prejudicar fontes legítimas; falsos negativos permitem que conteúdos desinformativos permaneçam sem sinalização. Uma faixa de revisão pode reduzir decisões indevidas, mas artigos abstidos precisam permanecer visíveis na avaliação.

Sistemas com abstenção devem reportar cobertura, taxa de revisão e erro entre os casos decididos. Não é válido comparar sua acurácia seletiva diretamente com a acurácia de um classificador que responde a todos os casos.

A precisão muda com a prevalência. Se, hipoteticamente, a taxa FAKE fosse 10%, mantendo recall de 95% e FPR de 5%, a precisão FAKE seria **0,95 × 0,10 / (0,95 × 0,10 + 0,05 × 0,90) ≈ 67,86%**. Desempenho em teste balanceado não garante a mesma precisão no tráfego real.

## 7. Principais desafios técnicos e estratégias de tratamento

### 7.1. Gerenciamento do ciclo de vida e MLOps

O desafio central é manter compatibilidade entre dados, pré-processamento, modelo e inferência. A coexistência da limpeza legada e facts-v2 exige contratos explícitos; aplicar a representação nova a pesos antigos pode invalidar predições.

Geração isolada de candidatos, manifestos e hashes constituem avanços concretos. Permanecem como evolução a integração DVC/MLflow, a reprodução em ambiente limpo e critérios de promoção que combinem qualidade, custo e risco. Seeds reduzem variação, mas não garantem identidade numérica entre hardwares e runtimes.

O retreino automático com votos foi desativado. Feedback comunitário deve alimentar uma fila de revisão, não substituir rótulos verificados. Após revisão, os dados podem gerar candidatos; a atualização do modelo servido depende de aprovação e possibilidade de rollback.

### 7.2. Integração entre serviços de nuvem

Uma arquitetura distribuída entre AWS e GCP introduz diferenças de identidade, permissões, armazenamento, rede e observabilidade. Transferências de corpus e pesos podem gerar custo e atraso; falhas parciais dificultam descobrir qual versão foi efetivamente usada.

O tratamento proposto envolve imagens portáveis, contratos HTTP versionados, hashes de artefatos, credenciais temporárias e rastreamento de requisições. Dados e inferência devem permanecer na mesma região e provedor quando isso atender aos requisitos. Treinamento em outro ambiente pode publicar um pacote imutável para implantação, evitando comunicação entre nuvens em cada predição.

Esses pontos são desafios de desenho e homologação; o repositório não demonstra operação multicloud validada.

### 7.3. Desbalanceamento e mudança de distribuição

Embora o corpus original seja balanceado, os recortes temporais exibem proporções diferentes. O tráfego pode variar por tema, fonte e período; também pode haver poucos exemplos representativos de certos grupos linguísticos.

As estratégias propostas incluem pesos de classe no LinearSVC, perda ponderada no transformer e amostragem somente no treino, comparadas com a baseline sem reponderação. O teste deve preservar a distribuição de interesse. Reamostrar antes da divisão produz risco de vazamento; interpolar vetores TF-IDF não garante exemplos linguisticamente válidos.

O monitoramento deve separar mudança dos dados de mudança da relação entre conteúdo e rótulo. Drift de entrada é um alerta; demonstrar degradação de classificação exige novos rótulos revisados. O monitor legado do projeto requer modernização antes de ser tratado como observabilidade operacional.

### 7.4. Latência e memória do mDeBERTa-v3

O transformer adiciona tokenização, cálculo neural e maior ocupação de memória. Extensão dos textos, número de hipóteses no zero-shot, batch, hardware e concorrência afetam o tempo de resposta. Inicialização dos pesos também pode comprometer disponibilidade durante a escala.

As medidas propostas são:

- provisionar pesos previamente, inicializar o modelo por processo e limitar concorrência;
- controlar tokens, usar padding dinâmico e avaliar janelas com agregação por artigo;
- testar batches e separar workers de inferência pesada da API quando necessário;
- investigar quantização, exportação e precisão reduzida, medindo regressão de qualidade;
- usar cache com chave de conteúdo, versão do modelo e pré-processamento, além de validade temporal;
- calcular explicações custosas sob demanda e avaliar cascata com revisão dos casos inconclusivos.

Não há benchmark comparável que sustente uma latência específica neste relatório. A homologação deve medir p50, p95 e p99, throughput, memória e taxa de erro em carga. Tokenização, inferência, extração web e busca externa devem ter medições separadas. Execuções frias e aquecidas precisam ser identificadas.

### 7.5. Qualidade factual e avaliação contemporânea

Um detector textual pode aprender atalhos relacionados a fonte, época ou tema. Mudanças superficiais de estilo podem enganá-lo sem alterar a alegação. Essa limitação motivou o protótipo de verificação por evidências e reforça a necessidade de teste recente anotado independentemente.

A avaliação deve incluir paráfrases, negações, números alterados, textos formais falsos, notícias verdadeiras com linguagem emotiva e casos com evidência insuficiente. Reputação de domínio e tom neutro contextualizam a análise, mas não substituem a verificação de cada alegação.

## 8. Aprendizados e lições adquiridas

### 8.1. Melhores práticas de MLOps

As evidências do desenvolvimento sustentam as seguintes lições:

1. **Reprodutibilidade depende do conjunto completo de artefatos.** Código, dados, splits, pré-processamento, dependências e pesos precisam estar vinculados.
2. **Ferramentas instaladas não comprovam adoção operacional.** Banco MLflow vazio ou ponteiro DVC sem remoto não demonstram rastreamento e recuperação completos.
3. **Candidatos devem ser separados da referência servida.** Essa prática permite auditar alterações e avaliar regressões antes da promoção.
4. **Treino e inferência precisam compartilhar contratos.** Correções de limpeza exigem retreinamento ou manutenção explícita da representação legada.
5. **Feedback é uma fonte de investigação.** Revisão independente protege contra ruído, manipulação e contaminação do corpus.

### 8.2. Pipelines escaláveis e escolha de modelos

A baseline linear estabelece uma referência de custo e desempenho que torna a avaliação de modelos maiores mais objetiva. A adoção de transformers deve decorrer de ganhos medidos no domínio de interesse, além dos resultados em benchmarks externos.

Pipelines modulares permitem evoluir extração, preparação, treino, avaliação e serving com responsabilidades claras. Escalabilidade também depende de limites de entrada, filas, cache, controle de concorrência e recuperação de falhas. A representação de texto deve preservar informação suficiente para a tarefa antes da busca por otimizações de hardware.

### 8.3. Impacto ético e social

Uma ferramenta de combate à desinformação pode ampliar a capacidade de triagem e apoiar educação midiática. Também pode produzir acusações indevidas, reforçar vieses do corpus ou induzir confiança excessiva em scores.

A prática responsável exige comunicar incerteza, apresentar fontes e permitir contestação. Avaliações por tema, período, fonte e variedade linguística ajudam a identificar danos assimétricos. Opinião, sátira, linguagem informal e sensacionalismo precisam ser tratados segundo critérios explícitos de anotação; não equivalem automaticamente a falsidade.

A revisão humana deve ser acionável e documentada, especialmente em casos de maior impacto. A utilidade social depende tanto da detecção de conteúdo problemático quanto da capacidade de preservar informação legítima e corrigir decisões equivocadas.

## 9. Conclusão e próximos passos

O projeto estabeleceu uma base técnica para classificação textual, explicabilidade linear e verificação por evidências. Os resultados históricos do candidato linear indicam capacidade discriminativa no corpus examinado, com limitações de origem, temporalidade e independência da avaliação.

A comparação hipotética ilustra um possível ganho do mDeBERTa-v3 ajustado, mas não autoriza sua promoção. A próxima etapa deve implementar o tracking MLflow, completar a reprodução com DVC, obter um corpus recente revisado e executar a comparação controlada. Em seguida, devem ser homologados os requisitos de latência, custo e implantação em AWS ou GCP.

O produto deve permanecer orientado ao apoio à análise: classificação textual, tom linguístico e verificação factual são resultados distintos, cada um com evidências e limites próprios.
