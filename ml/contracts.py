"""Shared bounded inputs and evidence provenance. Scores are not factual probabilities."""
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

LABELS = {0: "FAKE", 1: "REAL"}


def utcnow():
    return datetime.now(timezone.utc)


class NewsDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=100_000)
    language: Literal["pt"] = "pt"
    source_url: HttpUrl | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    event_date: date | None = None
    date_provenance: dict[Annotated[str, Field(max_length=40)], Annotated[str, Field(max_length=250)]] = Field(default_factory=dict, max_length=4)
    fetched_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def nonempty(self):
        if not self.body.strip():
            raise ValueError("Texto vazio")
        for field in ["published_at", "updated_at", "fetched_at"]:
            value = getattr(self, field)
            if value is not None and value.tzinfo is None:
                setattr(self, field, value.replace(tzinfo=timezone.utc))
        return self

    def content_hash(self):
        return hashlib.sha256(self.model_dump_json(exclude={"fetched_at"}).encode()).hexdigest()


class Verdict(str, Enum):
    SUPPORTED = "sustentada"
    REFUTED = "refutada"
    CONFLICTING = "evidencias_conflitantes"
    INSUFFICIENT = "evidencia_insuficiente"


class Claim(BaseModel):
    id: str
    text: str
    event_date: date | None = None
    # How event_date was obtained; publication date is never silently reused.
    event_date_provenance: str | None = None
    # Preceding title/sentence shown to search and reviewers; not part of the claim.
    context: str = ""
    checkable: bool = True
    limitations: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    claim_id: str
    url: HttpUrl
    title: str
    excerpt: str = Field(min_length=1, max_length=2_000)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    date_provenance: dict[str,str] = Field(default_factory=dict)
    accessed_at: datetime = Field(default_factory=utcnow)
    content_hash: str
    origin_group: str
    provider: str
    relation: Literal["sustenta", "refuta", "nao_conclusiva"] = "nao_conclusiva"
    relevance_score: float = Field(ge=0, le=1)
    score_kind: str = "cobertura_lexical_da_alegacao_nao_calibrada"
    # Descriptive only (e.g. public body domain); it never changes the relation or verdict.
    source_category: str = "nao_classificada"
    # How the URL was discovered: supplied URL or search provider, query, rank, listing date.
    retrieval: dict[str, str | int | None] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class ClaimResult(BaseModel):
    claim: Claim
    verdict: Verdict
    confidence: float | None = None
    confidence_kind: str = "indisponivel_sem_validacao_externa"
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str
    temporal_status: str
    limitations: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    system_version: str
    document: NewsDocument
    reference_date: date
    results: list[ClaimResult]
    summary: dict[str, int]
    partial: bool
    limitations: list[str]
    retrieval_status: str
    latency_ms: float
    cache_hit: bool = False
    auxiliary: dict = Field(default_factory=dict)
