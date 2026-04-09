import json
from typing import Dict, Any

class Evaluator:
    """Clase encargada de evaluar las predicciones de búsqueda y generación."""

    def evaluate(self, predictions_path: str, ground_truth_path: str, k: int) -> Dict[str, Any]:
        try:
            with open(predictions_path, 'r', encoding='utf-8') as f:
                predictions_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found - {predictions_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file - {predictions_path}")
            return {}

        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found - {ground_truth_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file - {ground_truth_path}")
            return {}

        if 'search_results' in predictions_data:
            predictions_list = predictions_data['search_results']
            mode = "search"
        else:
            predictions_list = predictions_data.get('answer_results', [])
            mode = "answer"

        gt_list = ground_truth_data.get('rag_questions', [])
        gt_dict = {item['question_id']: item for item in gt_list}

        total_questions = len(predictions_list)
        total_recall = 0.0
        
        evaluation_results = {
            "metrics": {
                "total_evaluated": total_questions,
                "k_used": k,
                "mode": mode
            },
            "detailed_results": []
        }

        for pred in predictions_list:
            q_id = pred['question_id']
            gt_item = gt_dict.get(q_id)
            
            if not gt_item:
                continue

            gt_sources = gt_item.get('sources', [])
            pred_sources = pred.get('retrieved_sources', [])[:k]
            
            hits = 0
            for gt_source in gt_sources:
                gt_file = gt_source.get('file_path')
                if any(p_source.get('file_path') == gt_file for p_source in pred_sources):
                    hits += 1
                    
            recall_at_k = (hits / len(gt_sources)) if gt_sources else 0.0
            total_recall += recall_at_k

            detail = {
                "question_id": q_id,
                "recall@k": recall_at_k
            }

            if mode == "answer":
                expected_answer = gt_item.get('answer', '')
                generated_answer = pred.get('answer', '')
                
                detail["expected_answer"] = expected_answer
                detail["generated_answer"] = generated_answer

            evaluation_results["detailed_results"].append(detail)

        mean_recall = (total_recall / total_questions) if total_questions > 0 else 0.0
        evaluation_results["metrics"]["mean_recall@k"] = round(mean_recall, 4)

        return evaluation_results