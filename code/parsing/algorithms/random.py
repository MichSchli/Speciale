import numpy as np

def fit(features, labels, model_path=None, save_every_iteration=False):
    pass

def predict(sentence_list, model_path=None):
    predictions = []
    for sentence in sentence_list:
        predictions.append([])
        for token_feature in sentence:
            predictions[-1].append(np.random.uniform(0.0, 1.0, len(sentence)))

    return predictions
