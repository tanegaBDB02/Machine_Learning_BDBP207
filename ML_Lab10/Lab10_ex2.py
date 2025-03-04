from Lab10_ex1 import calculate_entropy
import random

def information_gain(parent_labels, child_labels1, child_labels2):
    parent_entropy = calculate_entropy(parent_labels)
    total_samples = len(parent_labels)
    weight1 = len(child_labels1) / total_samples
    weight2 = len(child_labels2) / total_samples
    children_entropy = (weight1 * calculate_entropy(child_labels1)) + (weight2 * calculate_entropy(child_labels2))

    return parent_entropy - children_entropy

data = list(range(1, 11))
labels = [random.choice([0, 1]) for _ in data]

mid = len(data) // 2
data1, labels1 = data[:mid], labels[:mid]
data2, labels2 = data[mid:], labels[mid:]

print("Entropy for first split:", calculate_entropy(labels1))
print("Entropy for second split:", calculate_entropy(labels2))
print("Information Gain:", information_gain(labels, labels1, labels2))
