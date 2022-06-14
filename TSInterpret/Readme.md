# Target Structure 
- Model 
- Eval 
- Data 
- Explainer

# Other Todos
- [ ] Fix Lime 4 Time 
- [ ] pyts as normal installation 
- [ ] move datafile path  to config
- [ ] Structure Similar to Carla ? 
- [ ] Leftist solver via import 

# Evaluation of Time Series Models

Project to evaluate existing approaches to interpretability of time series data.
- [ ] Stream Setting 
  - https://arxiv.org/abs/2003.02544
  - https://jmread.github.io/talks/2019_04_09-Huawei_Workshop.pdf
## Significant Points Time Series
http://www.stat.ucla.edu/~frederic/221/W21/tsa4.pdf
https://www.kaggle.com/namgalielei/time-series-characteristics
- Statioanrity
  - Strict: mean and varaiance does not change over time 
  - Weak: A weakly stationary time series, x_t , is a finite variance process such that
(i) the mean value function, μ_t , defined in (1.9) is constant and does not depend on time t
(ii) the autocovariance function, γ(s, t), defined in (1.10) depends on s and t only through their difference |s − t|. [2]
  - Tests:  Augmented Dickey Fuller (ADF) test or Kwiatkowski–Phillips–Schmidt–Shin (KPSS)
- Trend 
- Sesonality



- Is there a trend, meaning that, on average, the measurements tend to increase (or decrease) over time?
- Is there seasonality, meaning that there is a regularly repeating pattern of highs and lows related to calendar time such as seasons, quarters, months, days of the week, and so on?
- Are there outliers? In regression, outliers are far away from your line. With time series data, your outliers are far away from your other data.
- Is there a long-run cycle or period unrelated to seasonality factors?
- Is there constant variance over time, or is the variance non-constant?
- Are there any abrupt changes to either the level of the series or the variance?

https://link.springer.com/chapter/10.1007/978-981-16-3264-8_15

## Analysis
https://hal.archives-ouvertes.fr/hal-01577883/file/TS_Review_Short.pdf
https://arxiv.org/pdf/2104.07406.pdf

- Indexing
- Clustering 
- Anomly Detection
- Change Point Detection 
- Motif Discovery
- Segmentation 
- Blind Source Seperation
- 
## Classification 
```
#Check SKTime for other Classifiaction Options: 

from sktime.registry import all_estimators

all_estimators(estimator_types="classifier", as_dataframe=True)
```
https://arxiv.org/abs/1809.04356
https://link.springer.com/article/10.1007/s10618-016-0483-9
- https://github.com/hfawaz/dl-4-tsc --> Paper for this
- Checkout: https://hpi.de/fileadmin/user_upload/fachgebiete/friedrich/documents/Schirneck/Mujkanovic_bachelors_thesis.pdf
-Classification
  - KNN 
  - HIVE COVE
  - Rework 
    - CNN
    - FFN
    - ResNet
- Time Series Forcasting
  - Time Delay Neural Network 
  - Long Short Term Memory
  - Gradient Boost Regressor
  - - ARIMA --> Continous 
- Anomely Detection
- The great Mutlivariate Time-series Bake-off : https://link.springer.com/article/10.1007/s10618-020-00727-3
## Data
Classification 

- 
- univariate : UCR Time-Series Repo
- The MIT-BIH Arrhythmia dataset [2]
- The PTB Diagnostic ECG database [3]
- mukltivariate: UCI Timeseries Repo
- Machine Anomoly Detection (Siddiqui et al. 2018)
- Mammography
- Cylinder-Bell-Funnel (CBF) benchmark dataset
- Nasa Shuttle
- http://timeseriesclassification.com/TSC.zip
- http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html
- HPAS Datsset
- Cori Dataset 
- NATOPS
- Taxomist Dataset 
Regression
- Rossmann store sales : https://www.kaggle.com/c/rossmann-store-sales
- Walamrt Recruitign Store Sales Forcasting : https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
- OhioT1Dm:http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
- Electricity: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

## Interpretability Methods
Use where possible preexisting work on time series , mark here if not possible. 
- [X] Lime 4 Time: https://github.com/emanuel-metzenthin/Lime-For-Time/blob/master/demo/LIME-Pipeline.ipynb --> How to cope if no probability as result ? 
- [ ] Shap: https://hpi.de/fileadmin/user_upload/fachgebiete/friedrich/documents/Schirneck/Mujkanovic_bachelors_thesis.pdf
- [X] GradCam: e.g. https://ai-fast-track.github.io/timeseries/cam_tutorial_ECG200/ ? :https://www.statworx.com/de/blog/erklaerbbarkeit-von-deep-learning-modellen-mit-grad-cam/
- [ ] CVAE
- [X] Counterfactual
- [X] Saliency: https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark
- [ ] Ancors
- [ ] LRP
- [ ] DeepLift 
- [X] Leftist : Agnostic local explanation for time series classification
- https://proceedings.neurips.cc/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf
- [ ] LSTM : http://iphome.hhi.de/samek/pdf/ArrXAI19.pdf , https://arxiv.org/pdf/1905.12034.pdf ,http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf 
- [ ] todo: https://github.com/h2oai/driverlessai-recipes 
## Evlauation Criteria
- [] Fron Evaluatzion of interpretability nethods for multivariate time series
  - Local Fidelity and Local Explanation 
    - Area over the pertubation Curve for Regression 
    - Ablation Percentage Threshold 
- General Explainability (https://arxiv.org/pdf/1806.07538.pdf) 
  - Explicitness : Are the explanations immediate and understandable?
  - Faithfulness :Are relevance scores indicative of "true" importance?
  - Stability :  How consistent are the explanations for similar/neighboring examples?
- Counterfactual Multivariate TS [Ates2020]
  - Qualitavly: Tree , Method and Lime SHap 
  - Comprehensability: Number of metrics
  - Faithfullness: Compare with Logistic Regression : 
    - Recall: How many of the metrics used by the classifier are in the explanation? 
    - Precision: How many of the metrics in the explanation are used by the classifier
  - Roboustness : Lipschitz constant
  - Generizability 
- Über Feature Space Assumption ?  [Mujkanovic et al . ]
- For Feature Importance/ Pertubation Based Models --> mostly Performance Evlauation  [Pravatharaju et al. 2021]
  - AUC: Difference Observe output if section with small / large feature importance is taken away
  - Confidence Supression Game: smallest number of time-steos required to supress black-box mpdel by x
  - Minimality: sum of saliency maps 
  - https://github.com/mlgig/explanation4tsc
  
## Tutorial & Libraries
- [ ] ts fresh: https://tsfresh.readthedocs.io/en/latest/text/forecasting.html
- [ ] https://www.machinelearningplus.com/time-series/time-series-analysis-python/
- [ ] https://www.kaggle.com/saurav9786/time-series-tutorial
- [ ] Outlier --> Pyod
- [ ] What does a LSTM remember ?
- smooth time-series, ... classification

# Architectural Concerns 
Something similar to :
![img.png](img.png)

# Submit to Neurips 
- Inspire by Carla