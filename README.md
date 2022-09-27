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

### Create Environment and Install Dependencies

```
$ conda env create --file environment.yml
$ conda activate VUS-env
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

## Experiments

|	|Precision@k|	Recall|	Precision|	Rrecall|	Rprecision|	F|	RF|	AUC_PR|	AUC_ROC|	R_AUC_PR|	R_AUC_ROC|	VUS_PR|	VUS_ROC|
|:--|:---------:|:-------:|:--------:|:-------:|:-----------:|:---:|:----:|:----:|:----:|:-----:|:----:|:----:|:----:|
|NormA|	4.210485|	4.249889|	4.787366|	4.559922|	4.463738|	4.425060|	4.650611|	4.253773|	4.103623|	4.298602|	4.379906|	4.293008|	4.300858|
|POLY|	5.384482|	4.903971|	4.994008|	5.109559|	4.855465|	4.920786|	5.006390|	4.686958|	4.704703|	4.535406|	5.050737|	4.473394|	4.983283|
|IForest|	5.042205|	5.114203|	5.075445|	5.849549|	4.820506|	5.103598|	5.547707|	4.540955|	4.301471|	4.570341|	4.406066|	4.621100|	4.406458|
|AE|	4.880552|	4.953687|	4.640731|	5.279224|	4.740862|	4.838507|	4.919577|	4.913290|	4.825540|	4.842853|	4.684716|	4.847660|	4.650359|
|OCSVM|	5.697530|	5.753513|	5.064816|	5.559130|	5.503605|	5.595893|	5.493684|	5.454006|	5.501606|	5.324205|	5.368112|	5.321574|	5.449086|
|MatrixProfile|	5.145945|	5.191028|	5.589128|	5.379395|	5.707388|	5.390321|	5.671893|	5.565779|	5.264788|	5.136523|	5.087060|	5.196917|	5.173278|
|LOF|	4.661508|	4.706821|	4.491874|	4.760564|	4.481798|	4.699444|	4.886699|	4.648609|	4.715578|	3.911382|	4.209517|	3.944675|	4.308522|
|LSTM|	5.089040|	5.163219|	5.363024|	4.345831|	5.339533|	5.122215|	4.496773|	5.705758|	6.162379|	6.581456|	6.348949|	6.559446|	6.288700|
|CNN|	4.888253|	4.963668|	4.993608|	4.156825|	5.087105|	4.904176|	4.326666|	5.230872|	5.420312|	5.799231|	5.464937|	5.742226|	5.439456|



## Usage

```python
import numpy as np
from src.models.feature import Window
from src.utils.metrics import metricor
from sklearn.preprocessing import MinMaxScaler
from src.analysis.robustness_eval import generate_curve


def anomaly_results(X_data):
    # Isolation Forest
    from src.models.iforest import IForest
    IF_clf = IForest(n_jobs=1)
    x = X_data
    IF_clf.fit(x)
    IF_score = IF_clf.decision_scores_

    return IF_score


def scoring(score):
    # Score normalization
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*np.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

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
X_data = Window(window = slidingWindow).convert(data).to_numpy()

if_score = anomaly_results(X_data)
for model_name, model_score in zip(names, [if_score]):
    print('Isolation Forest :', scoring(model_score))
```

