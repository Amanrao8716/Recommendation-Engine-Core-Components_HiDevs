import unittest

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator


class TestRecommendationEngine(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(SimilarityCalculator.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(SimilarityCalculator.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(SimilarityCalculator.cosine_similarity([], []), 0.0)
        self.assertEqual(SimilarityCalculator.cosine_similarity([0, 0], [1, 2]), 0.0)

    def test_jaccard_similarity(self):
        self.assertAlmostEqual(SimilarityCalculator.jaccard_similarity({"python", "java", "sql"}, {"python", "java", "spring"}), 2 / 4)
        self.assertEqual(SimilarityCalculator.jaccard_similarity({"a", "b"}, {"c", "d"}), 0.0)
        self.assertEqual(SimilarityCalculator.jaccard_similarity(set(), set()), 0.0)

    def test_pearson_correlation(self):
        self.assertAlmostEqual(SimilarityCalculator.pearson_correlation([5, 4, 3, 5], [4, 3, 2, 4]), 1.0)
        self.assertEqual(SimilarityCalculator.pearson_correlation([], []), 0.0)
        self.assertEqual(SimilarityCalculator.pearson_correlation([2, 2, 2], [4, 4, 4]), 0.0)

    def test_candidate_generator(self):
        users = {
            "user1": {"skills": {"python", "java", "sql"}, "history": ["item1", "item2"]},
            "user2": {"skills": {"python", "java", "spring"}, "history": ["item1", "item3"]},
            "user3": {"skills": {"python", "html", "css"}, "history": ["item2", "item4"]},
        }
        items = {
            "item1": {"tags": {"python", "programming"}, "popularity": 90},
            "item2": {"tags": {"java", "programming"}, "popularity": 80},
            "item3": {"tags": {"spring", "java", "backend"}, "popularity": 75},
            "item4": {"tags": {"html", "css", "frontend"}, "popularity": 70},
            "item5": {"tags": {"python", "machine-learning"}, "popularity": 95},
        }
        generator = CandidateGenerator(users, items, popularity={"item1": 90, "item2": 80, "item3": 75, "item4": 70, "item5": 95}, max_candidates=3)
        candidates = generator.hybrid_candidates("user1")
        self.assertTrue(len(candidates) <= 3)
        self.assertTrue("item3" in candidates or "item5" in candidates)

        new_user_candidates = generator.popularity_candidates()
        self.assertTrue(len(new_user_candidates) > 0)

    def test_scorer(self):
        scorer = RecommendationScorer()
        scorer.add_scorer("relevance", lambda user_id, item_id, context: 0.9, 0.5)
        scorer.add_scorer("popularity", lambda user_id, item_id, context: 0.8, 0.3)
        scorer.add_scorer("recency", lambda user_id, item_id, context: 0.7, 0.2)

        score = scorer.calculate_score("user1", "item1", {})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        ranked = scorer.rank_candidates("user1", ["item1", "item2", "item3"], 2)
        self.assertEqual(len(ranked), 2)

    def test_evaluator(self):
        recommendations = {
            "user1": ["item1", "item2", "item3"],
            "user2": ["item2", "item4"]
        }
        ground_truth = {
            "user1": ["item1", "item3"],
            "user2": ["item2", "item5"]
        }

        self.assertAlmostEqual(RecommendationEvaluator.precision_at_k(["item1", "item2", "item3"], ["item1", "item3"], 3), 2 / 3)
        self.assertAlmostEqual(RecommendationEvaluator.recall_at_k(["item1", "item2", "item3"], ["item1", "item3", "item5"], 3), 2 / 3)
        self.assertGreaterEqual(RecommendationEvaluator.ndcg_at_k(["item1", "item2", "item3"], ["item1", "item3"], 3), 0.0)

        metrics = RecommendationEvaluator.evaluate_all(recommendations, ground_truth, 3)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("ndcg", metrics)

    def test_integration(self):
        users = {
            "user1": {"skills": {"python", "java", "sql"}, "history": ["item1", "item2"]},
            "user2": {"skills": {"python", "java", "spring"}, "history": ["item1", "item3"]},
            "user3": {"skills": {"python", "html", "css"}, "history": ["item2", "item4"]},
        }
        items = {
            "item1": {"tags": {"python", "programming"}, "popularity": 90},
            "item2": {"tags": {"java", "programming"}, "popularity": 80},
            "item3": {"tags": {"spring", "java", "backend"}, "popularity": 75},
            "item4": {"tags": {"html", "css", "frontend"}, "popularity": 70},
            "item5": {"tags": {"python", "machine-learning"}, "popularity": 95},
        }

        generator = CandidateGenerator(users, items, popularity={"item1": 90, "item2": 80, "item3": 75, "item4": 70, "item5": 95}, max_candidates=3)
        candidates = generator.hybrid_candidates("user1")
        self.assertTrue(len(candidates) > 0)

        scorer = RecommendationScorer()
        scorer.add_scorer("relevance", lambda user_id, item_id, context: 0.8, 0.7)
        scorer.add_scorer("popularity", lambda user_id, item_id, context: 0.7, 0.3)
        ranked = scorer.rank_candidates("user1", candidates, 3)
        self.assertTrue(len(ranked) <= 3)

        gt = {"user1": ["item1", "item3"]}
        metrics = RecommendationEvaluator.evaluate_all({"user1": ranked}, gt, 3)
        self.assertIn("precision", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
