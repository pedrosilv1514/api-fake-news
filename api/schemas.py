"""Public API contracts. Frontend success responses retain the original flat keys."""
from datetime import date, datetime, timezone
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from ml.contracts import ClaimResult, VerificationResult

# A request example supplied by the user, not a claimed evidence or model result.
URL_EXAMPLE = (
    "https://g1.globo.com/rj/rio-de-janeiro/noticia/2026/10/08/"
    "mulher-descobre-que-seu-chico-colega-de-futebol-do-filho-e-chico-buarque.ghtml"
)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"url": URL_EXAMPLE}]})
    url: HttpUrl


class JanelaAnalise(BaseModel):
    evidencia: str
    qualidade_da_fonte: str
    corroboracao: str
    contexto: str
    atualidade: str


class AnalyzeResponse(BaseModel):
    # Nullable fields remain required keys, as in the original frontend output.
    url: str
    titulo: Optional[str]
    prob_fake: float
    taxa_confiabilidade: float
    bias_label: Optional[str]
    bias_score: Optional[float]
    janela_analise: JanelaAnalise
    o_que_sustenta: list[str]
    o_que_enfraquece: list[str]
    o_que_nao_foi_confirmado: list[str]
    modelos_usados: Optional[list[str]]
    score_kind: str = "similaridade_textual_calibrada_no_corpus_historico"
    confianca_factual: None = None
    aviso: str = (
        "prob_fake e taxa_confiabilidade representam um score textual histórico, "
        "não confiança factual. Evidências, quando disponíveis, ficam em verificacao."
    )


class VerificationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_urls: list[Annotated[HttpUrl, Field(max_length=2048)]] = Field(
        default_factory=list, max_length=5
    )
    reference_date: date = Field(default_factory=lambda: datetime.now(timezone.utc).date())
    search: bool = False

    @model_validator(mode="after")
    def temporal_reference(self):
        if self.reference_date > datetime.now(timezone.utc).date():
            raise ValueError("Referência futura não permitida")
        return self


class AnalyzeUrlRequest(VerificationOptions):
    """Frontend URL-only entry; metadata is extracted by the backend."""
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"url": URL_EXAMPLE}]}
    )
    url: Annotated[HttpUrl, Field(max_length=2048)]
    # This new endpoint attempts configured search by default. No default provider.
    search: bool = True


class VerificationDetails(BaseModel):
    """Factual namespace without duplicate article body or auxiliary scores."""
    system_version: str
    reference_date: date
    results: list[ClaimResult]
    summary: dict[str, int]
    partial: bool
    limitations: list[str]
    retrieval_status: str
    latency_ms: float
    cache_hit: bool

    @classmethod
    def from_result(cls, result: VerificationResult):
        return cls.model_validate(result.model_dump(exclude={"document", "auxiliary"}))


class FrontendAnalyzeResponse(AnalyzeResponse):
    verificacao: VerificationDetails


class FeedbackRequest(BaseModel):
    url: HttpUrl
    id_usuario: str = Field(min_length=1, max_length=200)
    voto: Literal[-1, 0, 1]  # Unreviewed feedback is never automatic ground truth.
