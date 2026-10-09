"""
Script de Avaliação Independente (Marco 4).
Avalia as métricas factuais (precisão, recall, etc.) sobre um corpus anotado.
AVISO: Executar sem um corpus validado por humanos não confere aprovação para produção.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import time
from fastapi.testclient import TestClient
from api.main import app

def main(dataset_path: Path, output_path: Path):
    if not dataset_path.exists():
        print(f"Dataset {dataset_path} não encontrado.")
        return 1

    lines = dataset_path.read_text().strip().splitlines()
    dataset = [json.loads(l) for l in lines]
    
    print(f"Carregados {len(dataset)} exemplos de {dataset_path}")
    print("ALERTA: Para aprovação final do Marco 4, este dataset deve conter notícias recentes reais anotadas de forma cega por humanos.")
    
    results = []
    confusion = {"sustentada": 0, "refutada": 0, "evidencias_conflitantes": 0, "evidencia_insuficiente": 0}
    correct = 0
    total = len(dataset)
    
    with TestClient(app) as client:
        for idx, item in enumerate(dataset):
            print(f"[{idx+1}/{total}] Avaliando: {item['text'][:50]}...")
            started = time.perf_counter()
            response = client.post("/v2/verificar", json={
                "document": {"body": item["text"]},
                "search": True
            })
            api_ms = (time.perf_counter() - started) * 1000
            
            if response.status_code != 200:
                print(f"Erro na API: {response.status_code}")
                predicted = "erro"
            else:
                data = response.json()
                if not data["results"]:
                    predicted = "evidencia_insuficiente"
                else:
                    # Toma o pior caso ou a primeira alegação
                    predicted = data["results"][0]["verdict"]
            
            ground_truth = item["label_factual"]
            if predicted == ground_truth:
                correct += 1
                
            if predicted in confusion:
                confusion[predicted] += 1
                
            results.append({
                "id": item.get("id"),
                "text": item["text"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "api_ms": round(api_ms, 2)
            })

    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "metrics": {
            "total": total,
            "accuracy": round(correct / max(1, total), 4),
            "confusion_predictions": confusion
        },
        "limitations": [
            "Dataset sintético/pequeno: não representa métricas de produção reais.",
            "Falta anotação humana independente de precisão >=95% e recall >=90%."
        ],
        "items": results
    }
    
    (output_path / "marco4_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\nResumo da Avaliação:")
    print(json.dumps(report["metrics"], indent=2))
    print(f"\nRelatório salvo em: {output_path / 'marco4_evaluation.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/marco4_sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/evolution_marco4"))
    args = parser.parse_args()
    main(args.dataset, args.output)
