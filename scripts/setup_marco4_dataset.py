import json
from pathlib import Path

data = [
    {
        "id": "hist_1",
        "url": "https://historico.local",
        "text": "Estão, finalmente, definidas as regras para produção da cannabis medicinal no Brasil. A ressalva não deslustra a ação da Agência Nacional de Vigilância Sanitária (Anvisa) ao tentar tirar o atraso do Brasil.",
        "label_factual": "sustentada",
        "notes": "Notícia histórica do banco_fake_news.csv. Espera-se que falhe em encontrar fontes no feed RSS recente, resultando em evidencia_insuficiente."
    },
    {
        "id": "sint_num",
        "url": "https://sintetico.local/1",
        "text": "O Banco Central anunciou hoje que a taxa Selic foi reduzida para 2,0% ao ano, contrariando as expectativas do mercado que aguardavam 10,5%.",
        "label_factual": "refutada",
        "notes": "Alegação numérica absurda e sintética recente. Espera-se refutação ou evidencia_insuficiente."
    },
    {
        "id": "sint_neg",
        "url": "https://sintetico.local/2",
        "text": "A equipe médica do hospital confirmou que o paciente não sobreviveu ao tratamento experimental.",
        "label_factual": "evidencia_insuficiente",
        "notes": "Sintético negativo genérico (sem âncoras fortes). Espera-se evidencia_insuficiente pois nenhuma busca trará confirmação desse contexto vago."
    }
]

file_path = Path("data/marco4_sample.jsonl")
with open(file_path, "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Dataset criado em {file_path} com {len(data)} registros.")
