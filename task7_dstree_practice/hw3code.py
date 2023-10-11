import numpy as np
from collections import Counter
import sklearn


def compute_bias_variance(regressor, dependence_fun, x_generator=np.random.uniform, noise_generator=np.random.uniform,
                          sample_size=300, samples_num=300, objects_num=200, seed=1234):
    np.random.seed(seed)
    x = x_generator(size=sample_size)
    noise = noise_generator(size=sample_size)
    samples = []
    noise_s = []
    for i in range(samples_num):
        xt = x_generator(size=sample_size)
        noise_st = noise_generator(size=sample_size)
        samples.append(xt)
        noise_s.append(noise_st)
    return compute_bias_variance_fixed_samples(regressor, dependence_fun, samples, x, noise_s, noise.mean())


def compute_bias_variance_fixed_samples(regressor, dependence_fun, samples, objects, noise, mean_noise):
    res = []
    for i in range(len(samples)):
        yt = dependence_fun(samples[i]) + noise[i]
        regressor.fit(samples[i][:, np.newaxis], yt)
        res.append(regressor.predict(objects[:, np.newaxis]))
    res = np.array(res)
    E = sum(res) / len(samples)
    bias = np.mean((E - (dependence_fun(objects) + mean_noise)) ** 2, axis=0)
    variance = np.mean(np.mean((res - E) ** 2, axis=0), axis=0)
    return bias, variance


def H(div):
    z = np.count_nonzero(div)
    nz = div.shape[0] - z
    z /= div.shape[0]
    nz /= div.shape[0]
    return 1 - z**2 - nz**2


def find_best_split(feature_vector, target_vector):
    m = np.array([feature_vector, target_vector])
    m = m.T
    m = m[m[:, 0].argsort()]
    m = m.T
    sort = m
    if (sort[0][0] == sort[0][-1]):
        return None
    sort2 = np.copy(sort)
    sort3 = np.copy(sort)
    sort3 = np.delete(sort3, 0, axis=1)
    sort2 = np.delete(sort2, -1, axis=1)
    thresholds = (sort3[0] + sort2[0]) / 2
    l = len(thresholds)
    ginis = np.zeros(l)
    for i in range(0, len(thresholds)):
        fg = i + 1
        l = sort[0].shape[0]
        left = -fg / l
        right = -(l - fg) / l
        ginis[i] = left * H(sort[1][:fg]) + right * H(sort[1][fg:])
    mi = np.argmax(ginis)
    return thresholds, ginis, thresholds[mi], ginis[mi]


class DecisionTree():
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._depth = None

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector[0] == feature_vector):
                continue

            best_split_found = find_best_split(feature_vector, sub_y)
            if best_split_found is None:
                continue

            _, _, threshold, gini = best_split_found
            if gini_best is None or gini > gini_best:
                a = feature_vector <= threshold
                if not ((len(sub_y[a]) == 0) or (len(sub_y[a]) == len(sub_y))):
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector <= threshold
                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0],
                                                  filter(lambda x: x[1] <= threshold, categories_map.items())))
                    else:
                        raise ValueError
        if(len(sub_y[split]) == len(sub_y)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]
        if (self._feature_types[feature_split] == "real" and x[feature_split] <= node["threshold"]) or \
                (self._feature_types[feature_split] == "categorical" and x[feature_split] in node["categories_split"]):
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
