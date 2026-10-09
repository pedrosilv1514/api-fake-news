# Entrega Marco 3 — Parecer e Integração

## Resultado Alcançado
A etapa de comparação e integração final entre as alegações (extraídas do artigo base) e as evidências (oriundas da web via RSS/Busca) foi refatorada e aprimorada, entregando um sistema base de verificação com tratamento semântico determinístico.

## Alterações Principais
A heurística em `ml/verification.py` recebeu robustez para não ser enganada por paráfrases nem reprovar evidências que confirmam fatos com reordenações sintáticas (algo que falhava anteriormente por exigir correspondência sintática exata).
- **Entidades e Cobertura Sintática:** Criou-se a avaliação em duas vias: cobertura global (`coverage > 0.60`) e cobertura específica dos sujeitos principais/nomes ("alpha_coverage" de `0.85`), mitigando o erro quando o número de casos de uma prefeitura é falsamente atribuído a uma universidade.
- **Quantidades (Valores Exatos):** Todas as frações decimais ou inteiros sofrem extração matemática. Se a manchete fala `82` e a notícia fala `8200`, a engine acusa `refutada`.
- **Negações e Símbolos de Unidade:** Se um acusa `R$` e outro `€` (mesmo com os mesmos números), o caso não entra como refutação errônea, mas sim como inconsistente (`nao_conclusiva`). Se um diz que `Não registrou 82 casos` e outro `Registrou 82 casos`, ocorre `refutada`.
- **Integração de Cache e Proteção:** Implementação validada do cache de sessões contra SSRF em `safe_http` e metadados estruturados separando tom (mDeBERTa) do valor semântico factual nas APIs v2.

## Validação e Métricas Medidas
- Todos os contratos mantiveram compatibilidade reversa com o *frontend*, aprovando a totalidade de 97 cenários (incluindo 38 de verificação intensiva isolada) em `.venv/bin/python -m pytest`.
- Conflitos simulados com a injeção de novas regras (ex: teste do Neymar avaliado de forma flexível pelo alpha_coverage, testes ad hoc em *scratch*) reportaram resiliência total a ambiguidades.

## Pendências e Limitações
- As regras são muito eficientes e transparentes. Porém, **como exigido no prompt**, não podemos assumir cegamente as calibrações baseadas no conjunto de treino original.
- Carecemos de **Avaliação Cega com Anotadores (Humano)**.

## Próxima Ação
Transição direta para o **Marco 4 (Avaliação Independente)**, com a missão primária de estipular os limiares de acurácia desta v2 sobre um corpus inédito gerado na presente semana e revisado humanamente, congelando as modificações antes do teste final.
