# Verificador de notícias em português

Estado em 08/10/2026: baseline textual corrigida e protótipo local de verificação
por evidências. Não homologado para produção; não há acurácia factual recente
validada nem modelo novo promovido.

Consulte o [registro de evolução](docs/EVOLUCAO_VERIFICADOR.md), a
[entrega técnica](docs/ENTREGA_EVOLUCAO_2026-10-08.md) e a
[investigação original](docs/INVESTIGACAO_MODELO_2026-10-08.md).

## Executar localmente

Python 3.12–3.14 é o intervalo declarado; os testes foram executados em 3.14.6.
Instale Poetry 2.5.1 e as dependências do lock:

~~~bash
poetry install --without bias
poetry run python -m nltk.downloader -d .cache/nltk_data stopwords
NLTK_DATA=.cache/nltk_data poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000
~~~

O download explícito de stopwords acima provisiona somente o classificador legado.
O startup não baixa pesos ou dados e não desabilita TLS. A rota /v2/verificar funciona
independentemente desse classificador; /ready mede a prontidão do componente
textual legado. /health informa separadamente classificador, tom e validação factual.

Tom via mDeBERTa é opcional (grupo bias e ENABLE_BIAS_MODEL=1), exige os pesos
da revisão fixada disponíveis localmente e não participa do parecer factual.

## API para o frontend

POST /analisar-url continua ativo, sem marca de depreciação, e mantém todos os
campos originais obrigatórios na raiz. O frontend existente não precisa migrar.

POST /v2/analisar-url recebe somente url como entrada obrigatória e retorna
os mesmos campos (prob_fake, taxa_confiabilidade, janela_analise, listas etc.),
com o parecer factual separado no objeto verificacao. Busca configurada é
tentada por padrão nessa nova rota; sem provedor há aviso e abstenção.
Nenhum corpo/metadado deve ser preenchido manualmente para analisar um link.

Veja o [contrato do frontend](docs/CONTRATO_FRONTEND_API.md), com entrada mínima,
tipos, tratamento de erros e diferenças de persistência/feedback. A nova rota
compatível requer o classificador textual, para não fabricar campos numéricos.

## API detalhada de evidências — auditoria

POST /v2/verificar recebe document ou url, exclusivamente. Exemplo local sem
fontes — intencionalmente inconclusivo:

~~~json
{
  "document": {
    "title": "Boletim municipal",
    "body": "A prefeitura registrou 82 casos em 06/10/2026.",
    "language": "pt"
  },
  "evidence_urls": [],
  "reference_date": "2026-10-08",
  "search": false
}
~~~

As URLs de notícia/evidências devem ser HTTPS públicas. Há até cinco fontes e
12 alegações por requisição. As conclusões são sustentada, refutada,
evidencias_conflitantes ou evidencia_insuficiente. Cada evidência traz URL,
trecho, publicação/atualização quando disponíveis, acesso e hash.
confidence é null: similaridade lexical não é probabilidade factual.

Forneça URLs reais em evidence_urls. Busca automática é opt-in (search=true),
via uma instância SearxNG HTTPS configurada em SEARXNG_BASE_URL; sem configuração
o sistema avisa e usa apenas as URLs fornecidas. Queries saem para essa instância:
não envie conteúdo sensível sem avaliar a política do provedor. A integração
foi testada com fixtures; uma instância real de busca ainda não foi homologada.

O endpoint /analisar-url permanece ativo com o contrato original:
prob_fake e taxa_confiabilidade são scores textuais históricos, não verdade
ou confiança factual. O cache legado tem TTL; o cache v2 usa conteúdo, fontes,
data de referência e versão, com TTL de cinco minutos. Fontes são baixadas
novamente antes de reutilizar um resultado v2.

POST /votar recebe feedback não revisado. Retreino automático a partir de votos
foi desativado para impedir contaminação e alteração do corpus de referência.

## Validar e gerar candidatos

~~~bash
poetry run python -m pytest -q
poetry run python -m ml.train
poetry run python -m scripts.validate_evolution --candidate models/candidates/candidate-20261008T195757552426Z
~~~

O treino gera um diretório novo em models/candidates, com manifesto, hashes,
splits e resultados. Nunca sobrescreve os dois modelos de referência.
Somente artefatos locais confiáveis podem ser carregados (joblib utiliza pickle).

O script opt-in python -m scripts.live_verification acessa duas notícias públicas
de outubro de 2026 e registra extração/API reais. Não é um teste cego nem mede
qualidade factual. Os dez exemplos locais do outro script são sintéticos.

Dockerfile e lock foram alinhados para startup offline sem o modelo de tom.
A imagem ainda não foi construída/testada: o daemon Docker local está indisponível.
docker compose up não deve ser interpretado como aprovação para deploy público.
Autenticação, rate limiting distribuído, revisão humana, monitoramento moderno,
avaliação independente e homologação/carga seguem pendentes.
