"""One presentation mapping for both frontend endpoints; factual data is separate."""
from api.schemas import (
    AnalyzeResponse, FrontendAnalyzeResponse, JanelaAnalise, VerificationDetails,
)
from ml.contracts import VerificationResult


def dict_to_response(
    url: str,
    titulo: str | None,
    prob_fake: float,
    bias_label: str | None,
    bias_score: float | None,
    xai_dict: dict,
    modelos_usados: list[str] | None,
) -> AnalyzeResponse:
    janela = xai_dict.get("janela_analise", {})
    return AnalyzeResponse(
        url=url,
        titulo=titulo,
        prob_fake=prob_fake,
        taxa_confiabilidade=round((1.0-prob_fake)*100, 2),
        bias_label=bias_label,
        bias_score=bias_score,
        janela_analise=JanelaAnalise(
            evidencia=janela.get("evidencia", ""),
            qualidade_da_fonte=janela.get("qualidade_da_fonte", ""),
            corroboracao=janela.get("corroboracao", ""),
            contexto=janela.get("contexto", ""),
            atualidade=janela.get("atualidade", ""),
        ),
        o_que_sustenta=xai_dict.get("o_que_sustenta", []),
        o_que_enfraquece=xai_dict.get("o_que_enfraquece", []),
        o_que_nao_foi_confirmado=xai_dict.get("o_que_nao_foi_confirmado", []),
        modelos_usados=modelos_usados,
    )


def frontend_response(
    requested_url: str, result: VerificationResult, prediction: dict
) -> FrontendAnalyzeResponse:
    base = dict_to_response(
        requested_url, result.document.title, prediction["prob_fake"],
        prediction["bias_label"], prediction["bias_score"],
        prediction.get("explicacao_xai") or {}, prediction["modelos_usados"],
    )
    return FrontendAnalyzeResponse(
        **base.model_dump(), verificacao=VerificationDetails.from_result(result)
    )

