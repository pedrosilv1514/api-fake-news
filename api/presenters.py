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
    prob_fake_original = prediction["prob_fake"]
    prob_fake_guardrail = prob_fake_original

    # Regra de Guardrail (Sobrescrita baseada em Verificação Factual)
    total_claims = sum(result.summary.values())
    if total_claims > 0:
        if result.summary.get("refutada", 0) > 0:
            # Se houver pelo menos uma alegação refutada, força a barra para 0% de confiabilidade (prob_fake = 1.0)
            prob_fake_guardrail = 1.0
            prediction["modelos_usados"].append("Guardrail: Refutada (Factual)")
        elif result.summary.get("evidencia_insuficiente", 0) == total_claims:
            # Se não conseguiu achar provas na internet, força para 50% de confiabilidade (prob_fake = 0.5)
            prob_fake_guardrail = 0.5
            prediction["modelos_usados"].append("Guardrail: Neutro/Insuficiente (Factual)")
        elif result.summary.get("sustentada", 0) > 0 and result.summary.get("refutada", 0) == 0 and result.summary.get("evidencias_conflitantes", 0) == 0:
            # Se houver provas sustentando e nada refutando/conflitando, confiança máxima
            prob_fake_guardrail = 0.0
            prediction["modelos_usados"].append("Guardrail: Sustentada (Factual)")

    base = dict_to_response(
        requested_url, result.document.title, prob_fake_guardrail,
        prediction["bias_label"], prediction["bias_score"],
        prediction.get("explicacao_xai") or {}, prediction["modelos_usados"],
    )
    return FrontendAnalyzeResponse(
        **base.model_dump(), verificacao=VerificationDetails.from_result(result)
    )

