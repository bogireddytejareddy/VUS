# Volume Under the Surface: new accuracy measures for abnormal subsequences detection in time series

The receiver operator characteristic (ROC) curve and the area under the curve (AUC) are widely used to compare the performance of different anomaly detectors. They mainly focus on point-based detection. However, the detection of collective anomalies concerns two factors: whether this outlier is detected and what percentage of this outlier is detected. The first factor is not reflected in the AUC. Another problem is the possible shift between the anomaly score and the real outlier due to the application of the sliding window. To tackle these problems, we incorporate the idea of range-based precision and recall, and suggest the range-based ROC and its counterpart in the precision-recall space, which provides a new evaluation for the collective anomalies. We finally introduce a new measure VUS (Volume Under the Surface) which corresponds to the averaged range-based measure when we vary the range size. We demonstrate in a large experimental evaluation that the proposed measures are significantly more robust to important criteria (such as lag and noise) and also significantly more useful to separate correctly the accurate from the the inaccurate methods.

<p align="center">
<img width="500" src="./docs/auc_volume.png"/>
</p>

## References

If you use VUS in your project or research, please cite our papers:

> John Paparrizos, Yuhao Kang, Paul Boniol, Ruey S. Tsay, Themis Palpanas,
and Michael J. Franklin. TSB-UAD: An End-to-End Benchmark Suite for
Univariate Time-Series Anomaly Detection. PVLDB, 15(8): 1697 - 1711, 2022.
doi:10.14778/3529337.3529354


> John Paparrizos, Paul Boniol, Themis Palpanas, Aaron Elmore,
and Michael J. Franklin. Volume Under the Surface: new accuracy measures for abnormal subsequences detection in time series. PVLDB, 15(X): X - X, 2022.
doi:X.X/X.X


## Data

To ease reproducibility, we share our results over [TSB-UAD](http://chaos.cs.uchicago.edu/tsb-uad/public.zip) benchmark dataset

## Installation

### Install Dependencies

```
$ conda env create --file environment.yml
$ conda activate VUS
$ pip install -r requirements.txt
```


### Install from [pip]()

```
$ pip install VUS
```

### Install from source
```
$ git clone https://github.com/johnpaparrizos/VUS
$ cd VUS/
$ python setup.py install
```

## Usage

```
import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.analysis.robustness_eval import generate_curve
from src.utils.metrics import metricor
from src.models.feature import Window


def anomaly_results(X_data):
    #Matrix Profile
    from src.models.matrix_profile import MatrixProfile
    MP_clf = MatrixProfile(window = slidingWindow)
    x = X_data
    MP_clf.fit(x)
    MP_score = MP_clf.decision_scores_

    #POLY
    from src.models.poly import POLY
    POLY_clf = POLY(power=3, window = slidingWindow)
    x = X_data
    POLY_clf.fit(x)
    POLY_score = POLY_clf.decision_scores_

    X_data = Window(window = slidingWindow).convert(X_data).to_numpy()

    # PCA
    from src.models.pca import PCA
    PCA_clf = PCA()
    x = X_data
    PCA_clf.fit(x)
    PCA_score = PCA_clf.decision_scores_

    # LOF
    from src.models.lof import LOF
    LOF_clf = LOF(n_neighbors=20, n_jobs=1)
    x = X_data
    LOF_clf.fit(x)
    LOF_score = LOF_clf.decision_scores_

    # Isolation Forest
    from src.models.iforest import IForest
    IF_clf = IForest(n_jobs=1)
    x = X_data
    IF_clf.fit(x)
    IF_score = IF_clf.decision_scores_

    return MP_score, POLY_score, PCA_score, LOF_score, IF_score


def scoring(score):
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
data = np.random.rand(5000)
labels = np.random.randint(2, size=5000)

names = ['Matrix Profile', 'PCA', 'LOF', 'IF']
mp_score, poly_score, pca_score, lof_score, if_score = anomaly_results(data)
for model_name, model_score in zip(names, [mp_score, pca_score, lof_score, if_score]):
    print(model_name + ':')
    scoring(model_score)
```

