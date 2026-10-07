class RecommendationScorer:
    def __init__(self):
        self.scorers = []

    def add_scorer(self, name, function, weight):
        if not callable(function):
            raise ValueError("Scorer function must be callable.")
        if weight < 0:
            raise ValueError("Weight must be non-negative.")
        self.scorers.append({"name": name, "function": function, "weight": float(weight)})

    def calculate_score(self, user_id, item_id, context):
        if not self.scorers:
            return 0.0

        total_weight = sum(scorer["weight"] for scorer in self.scorers)
        if total_weight == 0:
            return 0.0

        weighted_total = 0.0
        for scorer in self.scorers:
            try:
                score = float(scorer["function"](user_id, item_id, context or {}))
            except TypeError:
                score = float(scorer["function"](item_id, context or {}))
            score = max(0.0, min(1.0, score))
            weighted_total += score * scorer["weight"]

        final_score = weighted_total / total_weight
        return max(0.0, min(1.0, final_score))

    def rank_candidates(self, user_id, candidates, limit):
        if limit is None or limit <= 0:
            return []

        scored_candidates = []
        for item_id in candidates:
            score = self.calculate_score(user_id, item_id, {})
            scored_candidates.append((item_id, score))

        scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [item_id for item_id, _ in scored_candidates[:limit]]

    def explain_recommendation(self, user_id, item_id, context):
        if not context:
            return "Recommended based on a balanced relevance score."

        reasons = []
        if context.get("similar_users"):
            reasons.append("similar users")
        if context.get("popular"):
            reasons.append("high popularity")
        if context.get("match"):
            reasons.append("strong item match")
        if context.get("recent"):
            reasons.append("recent activity")

        if not reasons:
            return "Recommended based on overall relevance and ranking."

        return "Recommended because it matches your interests and is " + " and ".join(reasons) + "."
