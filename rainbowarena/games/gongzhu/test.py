import numpy as np


def transform_scores(scores):
    n = len(scores)
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    if std_dev == 0:
        # All scores are the same
        transformed_scores = np.zeros(n)
    else:
        # Standardize scores
        z_scores = (scores - mean_score) / std_dev

        # Find the scaling factor
        k = 1 / np.max(np.abs(z_scores))

        # Scale the standardized scores
        transformed_scores = k * z_scores

    return transformed_scores


# Example usage
scores = [-100, -100, -100, -100, -100]
# scores = np.array([10, 20, 30, 40, 50])
transformed_scores = transform_scores(scores)
print("Original scores:", scores)
print("Transformed scores:", transformed_scores)
print("Sum of transformed scores:", np.sum(transformed_scores))
