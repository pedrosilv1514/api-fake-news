# Entrega Marco 4 — Avaliação Independente (Parcial)

## Resultado Alcançado
A base de testes para avaliação independente sobre dados reais (live) foi estabelecida. Foram construídos os roteiros automáticos para mensuração de F1, Precisão, Recall e FPR sobre o pipeline v2 que executa verificação ativa na internet.

## Alterações Principais
- Inserido o script de avaliação `scripts/evaluate_marco4.py` projetado para rodar contra um dataset serializado de notícias contemporâneas.
- Inicializado o arquivo `data/marco4_sample.jsonl` com o esquema JSON rigoroso que suporta Ground Truth ("label_factual").

## Lacuna Registrada: Bloqueio de Promoção
De acordo com o *Prompt Mestre*, "se faltar anotação humana, registre a lacuna sem fabricar aprovação". 
Sendo um desenvolvimento contínuo sem integração de avaliadores humanos no exato momento, informamos a impossibilidade de reportar as métricas oficiais de `precisão >= 95%` e `recall >= 90%` no corpus contemporâneo. O teste ad-hoc revelou que, sob regras estritas de segurança (Marco 3), a abstenção (`evidencia_insuficiente`) se torna muito prevalente em casos pequenos (resultando provisoriamente em acurácia próxima de zero no cenário cego minúsculo e não-curado).

## Próxima Ação
A equipe editorial e de MLOps necessita injetar um lote real de afirmações recentes e verificadas da última semana no arquivo `marco4_sample.jsonl`.
Assim que preenchido, o comando `.venv/bin/python scripts/evaluate_marco4.py` fornecerá o sinal verde para certificar que o identificador de fato superou as metas estatísticas, autorizando o início do **Marco 5 (Homologação e Operação)** e seu consequente *Deploy*.
