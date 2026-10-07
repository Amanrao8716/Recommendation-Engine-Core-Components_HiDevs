import math


class RecommendationEvaluator:
    @staticmethod
    def precision_at_k(recommendations, relevant_items, k):
        if k <= 0 or not recommendations:
            return 0.0

        top_k = list(recommendations)[:k]
        relevant_set = set(relevant_items or [])
        hits = len(set(top_k).intersection(relevant_set))
        return hits / float(k)

    @staticmethod
    def recall_at_k(recommendations, relevant_items, k):
        relevant_set = set(relevant_items or [])
        if not relevant_set:
            return 0.0

        top_k = list(recommendations)[:k]
        hits = len(set(top_k).intersection(relevant_set))
        return hits / float(len(relevant_set))

    @staticmethod
    def ndcg_at_k(recommendations, relevant_items, k):
        if k <= 0 or not recommendations:
            return 0.0

        relevant_set = set(relevant_items or [])
        if not relevant_set:
            return 0.0

        dcg = 0.0
        for index, item in enumerate(recommendations[:k], start=1):
            if item in relevant_set:
                dcg += (2 ** 1 - 1) / math.log2(index + 1)

        ideal_count = min(len(relevant_set), k)
        idcg = 0.0
        for index in range(1, ideal_count + 1):
            idcg += (2 ** 1 - 1) / math.log2(index + 1)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def evaluate_all(recommendations_dict, ground_truth_dict, k):
        if not recommendations_dict:
            return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}

        valid_users = []
        for user_id, recommendations in recommendations_dict.items():
            if user_id not in ground_truth_dict:
                continue
            valid_users.append((user_id, recommendations, ground_truth_dict[user_id]))

        if not valid_users:
            return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}

        precision_total = 0.0
        recall_total = 0.0
        ndcg_total = 0.0

        for _, recommendations, relevant_items in valid_users:
            precision_total += RecommendationEvaluator.precision_at_k(recommendations, relevant_items, k)
            recall_total += RecommendationEvaluator.recall_at_k(recommendations, relevant_items, k)
            ndcg_total += RecommendationEvaluator.ndcg_at_k(recommendations, relevant_items, k)

        total_users = len(valid_users)
        return {
            "precision": precision_total / total_users,
            "recall": recall_total / total_users,
            "ndcg": ndcg_total / total_users,
        }
