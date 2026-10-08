# Contrato de integração com o frontend

Atualizado em 08/10/2026. Os campos originais continuam na raiz e são
obrigatórios em respostas HTTP 200; detalhes factuais ficam separados.

## Qual endpoint consumir

| Endpoint POST | Uso | Resposta |
| --- | --- | --- |
| /analisar-url | Frontend existente, sem migração | Campos originais + avisos |
| /v2/analisar-url | Frontend com evidências | Mesmos campos originais + verificacao |
| /v2/verificar | Auditoria detalhada, texto ou URL | Contrato técnico de alegações |

/analisar-url é ativo e não está mais marcado como depreciado.
A rota detalhada continua disponível sem mudar sua resposta.

## Entrada mínima

Para as duas rotas de análise, envie apenas url na raiz, sem preencher título,
corpo ou datas. Exemplo com o link fornecido durante o desenvolvimento
(não é promessa de extração bem-sucedida desse portal):

~~~json
{
  "url": "https://g1.globo.com/rj/rio-de-janeiro/noticia/2026/10/08/mulher-descobre-que-seu-chico-colega-de-futebol-do-filho-e-chico-buarque.ghtml"
}
~~~

/v2/analisar-url extrai a notícia e tenta a busca configurada por padrão.
Sem SEARXNG_BASE_URL, informa search_not_configured e não inventa evidências.
Nenhum provedor foi configurado/contratado nesta entrega. Opções adicionais:
evidence_urls (até cinco), reference_date e search. Use search=false para
desabilitar a busca. Consultas saem para a instância escolhida pelo operador.

/v2/verificar mantém busca opt-in e aceita document ou url, exclusivamente.
O Swagger agora oferece exemplos separados por modo, sem fornecer ambos
simultaneamente ou incluir endereços de exemplo como evidências.

## Campos obrigatórios de sucesso

Nomes e tipos originais foram preservados:

~~~typescript
interface AnaliseOriginal {
  url: string;
  titulo: string | null;
  prob_fake: number;
  taxa_confiabilidade: number;
  bias_label: string | null;
  bias_score: number | null;
  janela_analise: {
    evidencia: string;
    qualidade_da_fonte: string;
    corroboracao: string;
    contexto: string;
    atualidade: string;
  };
  o_que_sustenta: string[];
  o_que_enfraquece: string[];
  o_que_nao_foi_confirmado: string[];
  modelos_usados: string[] | null;
}
~~~

Todos aparecem como required no OpenAPI das duas respostas. Nullable não é
chave ausente: bias_score=null significa tom indisponível, não score zero.
As listas e os cinco campos da janela permanecem disponíveis.

Campos adicionais score_kind, confianca_factual e aviso explicitam a semântica.
prob_fake/taxa_confiabilidade mantêm a fórmula original e são scores textuais,
não probabilidade de a notícia ser verdadeira. confianca_factual permanece null.
Os campos o_que_sustenta/o_que_enfraquece na raiz continuam sendo XAI textual.

## Evidências sem alterar a raiz

Somente a nova rota compatível acrescenta verificacao, contendo versão,
referência, resumo das quatro conclusões, resultados por alegação,
justificativas, temporalidade, evidências, limitações, status de busca/cache
e latência local. Documento completo e scores auxiliares não são duplicados.
O frontend existente pode renderizar a raiz; novos componentes podem consumir
verificacao separadamente. Esquema completo em /openapi.json.

Cada evidência mantém URL, trecho, datas, acesso, hash e procedência.
confidence permanece null e pertinência não é probabilidade factual.
Mesmo um parecer refutado/conflitante não sobrescreve o score textual.
latency_ms da nova rota inclui coleta e inferência, mas não fila, transporte
público ou banco.

## Erros e banco

A obrigação de campos vale para HTTP 200. Trate detail em respostas
400/422/429/503 antes de acessar a análise. Sem classificador, a nova rota
retorna 503, nunca números inventados. Extração depende de HTTPS público:
bloqueios de portal, paywall ou ausência de texto extraível continuam possíveis.

A rota original mantém cache/banco e a vinculação a /votar. A nova rota é
independente do banco legado e não cria registros para votação. Não migre
feedback para ela sem implementar/testar essa persistência; manter a rota
original preserva o fluxo existente. Feedback ainda exige revisão.

Reinicie a API se não houver auto-reload e atualize /docs. Não houve deploy,
retreino, promoção de modelo ou validação de provedor real neste ajuste.

