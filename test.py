import math
import numpy as np
import pandas as pd
from vus.models.feature import Window
from vus.utils.metrics import metricor
from sklearn.preprocessing import MinMaxScaler
from vus.analysis.robustness_eval import generate_curve


def anomaly_results(X_data):
    # Isolation Forest
    from vus.models.iforest import IForest
    IF_clf = IForest(n_jobs=1)
    x = X_data
    IF_clf.fit(x)
    IF_score = IF_clf.decision_scores_

    return IF_score


def scoring(score, labels, slidingWindow):
    # Score normalization
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    # Computing RANGE_AUC_ROC and RANGE_AUC_PR
    grader = metricor()
    R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)

    # Computing VUS_ROC and VUS_PR
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score,2*slidingWindow)
    print(R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR)


# Data Preprocessing
slidingWindow = 100 # user-defined subsequence length
dataset = pd.read_csv('./data/MBA_ECG805_data.out', header=None).to_numpy()
data = dataset[:, 0]
labels = dataset[:, 1]
X_data = Window(window = slidingWindow).convert(data).to_numpy()

if_score = anomaly_results(X_data)
print('Isolation Forest :')
scoring(if_score, labels, slidingWindow)
