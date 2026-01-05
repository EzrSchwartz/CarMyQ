import collections
from resnetclassification import classify
def majority_vote(frame_classes, window=10, threshold=0.7):
    if len(frame_classes) < window:
        return None  # not enough frames yet
    window_classes = frame_classes[-window:]
    counts = collections.Counter(window_classes)
    top_class, count = counts.most_common(1)[0]
    percent = count / window
    return top_class if percent >= threshold else None


def predicitons(frame):
    class_idx = classify(frame)
    return class_idx