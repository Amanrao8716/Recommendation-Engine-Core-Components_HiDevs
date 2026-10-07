from similarity import SimilarityCalculator


class CandidateGenerator:
    def __init__(self, users=None, items=None, popularity=None, max_candidates=10):
        self.users = users or {}
        self.items = items or {}
        self.popularity = popularity or {}
        self.max_candidates = max_candidates

    def _limit(self, candidates):
        if self.max_candidates <= 0:
            return []
        return list(candidates)[: self.max_candidates]

    def collaborative_candidates(self, user_id):
        if user_id not in self.users:
            return self.popularity_candidates()

        target_user = self.users.get(user_id, {})
        target_history = set(target_user.get("history", []))
        target_skills = set(target_user.get("skills", set()))

        similar_users = []
        for other_id, data in self.users.items():
            if other_id == user_id:
                continue
            other_skills = set(data.get("skills", set()))
            similarity = SimilarityCalculator.jaccard_similarity(target_skills, other_skills)
            if similarity > 0:
                similar_users.append((similarity, other_id, data))

        similar_users.sort(key=lambda item: item[0], reverse=True)

        candidates = []
        for _, other_id, data in similar_users:
            for item_id in data.get("history", []):
                if item_id in target_history:
                    continue
                if item_id not in self.items:
                    continue
                candidates.append(item_id)

        seen = set()
        ordered = []
        for item_id in candidates:
            if item_id not in seen:
                seen.add(item_id)
                ordered.append(item_id)

        return self._limit(ordered)

    def content_based_candidates(self, user_id):
        if user_id not in self.users:
            return self.popularity_candidates()

        target_user = self.users.get(user_id, {})
        histories = target_user.get("history", [])
        if not histories:
            return self.popularity_candidates()

        candidate_scores = []
        for item_id, item_data in self.items.items():
            if item_id in histories:
                continue
            item_tags = set(item_data.get("tags", set()))
            if not item_tags:
                continue

            score = 0.0
            for history_item in histories:
                history_data = self.items.get(history_item, {})
                history_tags = set(history_data.get("tags", set()))
                if not history_tags:
                    continue
                score += SimilarityCalculator.jaccard_similarity(history_tags, item_tags)

            if histories:
                score /= len(histories)
            candidate_scores.append((score, item_id))

        candidate_scores.sort(key=lambda pair: pair[0], reverse=True)
        ordered = [item_id for _, item_id in candidate_scores]
        return self._limit(ordered)

    def popularity_candidates(self):
        ordered = sorted(self.popularity.items(), key=lambda pair: pair[1], reverse=True)
        return self._limit([item_id for item_id, _ in ordered])

    def hybrid_candidates(self, user_id):
        combined = []

        collaborative = self.collaborative_candidates(user_id)
        content_based = self.content_based_candidates(user_id)
        popular = self.popularity_candidates()

        for source in (collaborative, content_based, popular):
            for item_id in source:
                combined.append(item_id)

        seen = set()
        ordered = []
        for item_id in combined:
            if item_id not in seen:
                seen.add(item_id)
                ordered.append(item_id)

        if not ordered and self.popularity:
            return self.popularity_candidates()

        return self._limit(ordered)
