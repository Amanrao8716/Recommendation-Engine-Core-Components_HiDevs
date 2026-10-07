import math


class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(vec1, vec2):
        if not vec1 or not vec2:
            return 0.0

        vec1 = list(vec1)
        vec2 = list(vec2)
        length = min(len(vec1), len(vec2))
        if length == 0:
            return 0.0

        dot = sum(float(a) * float(b) for a, b in zip(vec1[:length], vec2[:length]))
        norm1 = math.sqrt(sum(float(v) ** 2 for v in vec1[:length]))
        norm2 = math.sqrt(sum(float(v) ** 2 for v in vec2[:length]))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def jaccard_similarity(set1, set2):
        left = set(set1)
        right = set(set2)

        if not left and not right:
            return 0.0

        union = left.union(right)
        if not union:
            return 0.0

        intersection = left.intersection(right)
        score = len(intersection) / len(union)
        return max(0.0, min(1.0, score))

    @staticmethod
    def pearson_correlation(ratings1, ratings2):
        if not ratings1 or not ratings2:
            return 0.0

        ratings1 = [float(v) for v in ratings1]
        ratings2 = [float(v) for v in ratings2]
        length = min(len(ratings1), len(ratings2))
        if length == 0:
            return 0.0

        x = ratings1[:length]
        y = ratings2[:length]

        mean_x = sum(x) / length
        mean_y = sum(y) / length

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator_x = sum((xi - mean_x) ** 2 for xi in x)
        denominator_y = sum((yi - mean_y) ** 2 for yi in y)

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        score = numerator / math.sqrt(denominator_x * denominator_y)
        return max(-1.0, min(1.0, score))
