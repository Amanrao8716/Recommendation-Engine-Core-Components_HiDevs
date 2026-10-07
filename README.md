# Recommendation Engine Core Components

## Overview
This project demonstrates a small recommendation engine built entirely with Python data structures and rule-based logic. It shows the full recommendation workflow: measuring similarity, generating candidate items, scoring and ranking recommendations, and evaluating the quality of the results.

The project intentionally avoids databases and machine-learning models. Instead, it uses in-memory dictionaries, sets, lists, and simple algorithms so the system remains easy to understand and test.

## Architecture
The project follows this flow:

Similarity
→ Candidate Generation
→ Scoring
→ Ranking
→ Evaluation

This mirrors how real recommendation systems work at a simplified level:

1. Compare users or items by similarity.
2. Generate a list of possible recommendations.
3. Score each candidate using weighted rules.
4. Rank the candidates by final score.
5. Evaluate recommendations using precision, recall, and NDCG.

## Components

### similarity.py
Contains the `SimilarityCalculator` class with methods for:
- `cosine_similarity(vec1, vec2)`
- `jaccard_similarity(set1, set2)`
- `pearson_correlation(ratings1, ratings2)`

### candidate_gen.py
Contains the `CandidateGenerator` class with methods for:
- `collaborative_candidates(user_id)`
- `content_based_candidates(user_id)`
- `popularity_candidates()`
- `hybrid_candidates(user_id)`

### scorer.py
Contains the `RecommendationScorer` class with:
- `add_scorer(name, function, weight)`
- `calculate_score(user_id, item_id, context)`
- `rank_candidates(user_id, candidates, limit)`
- `explain_recommendation(user_id, item_id, context)`

### evaluator.py
Contains the `RecommendationEvaluator` class with:
- `precision_at_k(recommendations, relevant_items, k)`
- `recall_at_k(recommendations, relevant_items, k)`
- `ndcg_at_k(recommendations, relevant_items, k)`
- `evaluate_all(recommendations_dict, ground_truth_dict, k)`

### test.py
Includes unit tests covering the core functionality, edge cases, and an integration flow connecting the full system.

## Similarity Algorithms

### Cosine Similarity
Cosine similarity compares the angle between two vectors.

Formula:

cosine_similarity = (vec1 · vec2) / (||vec1|| × ||vec2||)

It is useful when comparing numeric user or item vectors. The implementation handles empty vectors, zero-length vectors, and zero vectors safely.

### Jaccard Similarity
Jaccard similarity compares the overlap between two sets.

Formula:

J(A, B) = |A ∩ B| / |A ∪ B|

This is useful for matching user skills, tags, genres, or item categories.

### Pearson Correlation
Pearson correlation measures the linear relationship between two rating sequences.

Values range from -1 to 1:
- +1: strong positive correlation
- 0: no linear relationship
- -1: strong negative correlation

The implementation handles empty values, different lengths, and constant inputs without crashing.

## Recommendation Strategies

### Collaborative Filtering
This strategy finds users with similar tastes and recommends items those similar users have liked.

### Content-Based Recommendation
This strategy recommends items similar to what the target user has already interacted with, using metadata like tags or categories.

### Popularity-Based Recommendation
This strategy ranks the most popular items overall as a fallback, especially for new or cold-start users.

### Hybrid Recommendation
The hybrid strategy combines collaborative filtering, content-based filtering, and popularity to make the candidate list more robust.

## Scoring
The scorer accepts multiple scoring functions and combines them using weights.

Example:

Final Score = (relevance × 0.5) + (popularity × 0.3) + (recency × 0.2)

Each scoring function contributes according to its weight, and the final result is clamped between 0 and 1.

## Evaluation
The project includes three common recommendation metrics:

### Precision@K
Measures how many recommended items are relevant.

### Recall@K
Measures how many relevant items were actually recommended.

### NDCG@K
Measures ranking quality by rewarding relevant items that appear near the top of the list.

The evaluator also supports multiple users and skips users missing ground-truth data to avoid errors in sparse datasets.

## How to Run
From the project directory:

```bash
python test.py
```

## Sample Output

```text
================================
RECOMMENDATION ENGINE TESTS
================================

Testing Similarity Calculator...
✓ Cosine similarity passed
✓ Jaccard similarity passed
✓ Pearson correlation passed

Testing Candidate Generator...
✓ Collaborative candidates passed
✓ Content-based candidates passed
✓ Popularity candidates passed
✓ Hybrid candidates passed

Testing Scorer...
✓ Weighted scoring passed
✓ Ranking passed

Testing Evaluator...
✓ Precision@K passed
✓ Recall@K passed
✓ NDCG@K passed

================================
ALL TESTS PASSED
================================
```

## Limitations
This project uses:
- in-memory dictionaries
- small sample datasets
- simple rule-based scoring
- no machine-learning model
- no database

## Future Improvements
Possible next steps include:
- real databases and persistent storage
- larger datasets
- collaborative filtering with more advanced similarity methods
- machine-learning models or embeddings
- API or web interface
- user feedback and personalization
- real-time recommendation updates

## Missing Ground Truth Behavior
When a user has no ground-truth data, the evaluator skips that user and continues evaluating the remaining users. This avoids crashes and keeps the evaluation process stable for sparse or partially labeled datasets.
