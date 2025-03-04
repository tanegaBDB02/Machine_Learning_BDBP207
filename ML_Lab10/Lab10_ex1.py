import math

def calculate_entropy(class_labels):
    total_items = len(class_labels)
    class_counts = {}
    for label in class_labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    entropy = 0.0

    for count in class_counts.values():
        probability = count / total_items
        entropy -= probability * math.log2(probability)

    return entropy

class_labels = ['A', 'A', 'B', 'A', 'B', 'B', 'A', 'A', 'A', 'B']
entropy = calculate_entropy(class_labels)
print("Entropy of the dataset:", entropy)