# Entrega Marco 2 — Evidências Atuais

## Resultado Alcançado
A capacidade de coletar evidências atuais de fontes públicas para verificar alegações extraídas foi implementada e validada com sucesso, mantendo total interoperabilidade com os contratos do Marco 1 e do Frontend. Foi estipulado um fluxo robusto para desvincular deduplicação de domínios, temporalidade de fatos (eventos vs. datas de publicação), com rastreabilidade completa das proveniências de recuperação, e tolerância a falhas na busca via provedores de resguardo.

## Alterações Principais
- **Categorização e Agrupamento (`ml/sources.py`)**: As fontes agora são categorizadas por nível de autoridade (como instâncias públicas brasileiras) e agrupadas pelo domínio raiz para prevenir que publicações redundantes de uma mesma rede atuem como confirmação independente.
- **Consultas e Pertinência (`ml/terms.py`)**: Implementado sistema de construção de query a partir da alegação para realizar buscas abertas mais objetivas nos buscadores, bem como cálculo de pertinência lexical que filtra trechos menos aderentes na hora do ranqueamento das evidências.
- **Tratamento Temporal (`ml/temporal.py`)**: As datas relativas (ex: "hoje", "sexta passada") citadas no corpo da notícia são referenciadas e resolvidas contra a âncora da data de publicação do documento original, propiciando um rastreio da data exata do evento (`event_date`).
- **Resolução de Provedor de Busca Pública (`scraper/providers.py` e `api/routers/verificacao.py`)**: Implementado o provedor `BingNewsRssProvider` servindo como alternativa livre quando instâncias customizadas do SearxNG estiverem desligadas. Links originais da Microsoft contendo `apiclick.aspx` são parseados e sanitizados em HTTPS para que o robô faça a extração segura do conteúdo de destino, sem bloqueios de segurança (`validate_url`).

## Validação e Métricas Medidas
- Suíte Pytest (97 casos de teste) rodando de forma limpa, confirmando regressão zero após ajustes na fixture `NoSearchProvider`.
- Extração controlada via `scripts/live_verification.py` confirmada.
- Busca live e end-to-end de alegação via `api/routers/verificacao.py` efetuada e confirmada via chamada cliente-servidor para `POST /v2/verificar`. No teste real, uma alegação esportiva sintética usou termos da query no BingNews, identificou matéria do site Infomoney, efetuou o fetch HTTPS limpo, leu metadados meta (og:updated_time, article:published_time), checou similaridade e categorizou a predição como evidência insuficiente face aos pesos semânticos ausentes da regra conservadora.

## Pendências e Limitações
- Concordância sintática pura ainda barra certas equivalências semânticas. O motor de comparação lexical ainda é restrito (relevance_score).
- Depende da calibração factual independente via testadores humanos sobre notícias contemporâneas para avançar os pesos numéricos de veredito de fontes não governamentais.

## Próxima Ação
Retomar o Marco 3 focado em **Parecer e Integração**, ou transitar para a anotação humana exigida para o **Marco 4 (Avaliação Independente)**, mantendo sempre candidatos já persistidos sem substituí-los ou afetar a operação do modelo atual que já serve as pontas no endpoint legado.
