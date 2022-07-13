# Algorithm Overview
This page provides a high level overview on algorithms and their features currently implemented in TSInterpret.


Method       | Backend       | Type         | Data          |Training Set
------------ | ------------- | ------------ | ------------- | ------------
COMTE [@ates_counterfactual_2021] | TF,PYT, SK, BB  | InstanceBased | multi  | y
LEFTIST [@guilleme_agnostic_2019] | TF,PYT, SK, BB | FeatureAttribution | uni  | y
NUN-CF [@sanchez-ruiz_instance-based_2021] | TF,PYT, SK  | InstanceBased | uni  | y
TSR [@ismail_benchmarking_2020] | TF,PYT  | FeatureAttribution | multi  | n

<p> <small><b>BB</b>: black-box, <b>TF</b>: tensorflow, <b>PYT</b>: PyTorch, <b>SK</b>: sklearn, <b>uni</b>: univariate time series, <b>multi</b>: multivariate time series, <b>y</b>: yes, <b>n</b>: no </small></p>

<b>NUN-CF</b> Delany et al.[@sanchez-ruiz_instance-based_2021] proposed using a native guide to generate coun-
terfatuals. The native guide is selected via K-nearest neighbor. For generating the
counterfactual they offer three options: the native guide, the nativw guide with bary
centering, and the counterfactual based on the native guide and class activation map-
ping.

<b>COMTE</b> Ates et al. [@ates_counterfactual_2021] proposed COMTE as a heuristic pertubations approach
for multivariate time series counterfactuals. The goal is to change a small number of
features to obtain a counterfactual.

<b>LEFTIST</b> Agnostic Local Explanation for Time Series Classification by Guilleme et al.[@guilleme_agnostic_2019]
adapted LIME for time series classification task and propose to use prefixed (both the
length and the position) shapelets as the interpretable components, and provide the
feature importance of each shapelet.

<b>TSR</b> Temporal Saliency Rescaling Ismail et al. [@ismail_benchmarking_2020] calculates the importance of each
timestep, followed by the feature importance on basis of different Saliency Methods,
both Back-propagation based and perturbation based. For a full list of implemented
methods, we refer the reader to our code documentation. The implementation in TSIn-
terpret is based on tf-explain [@meudec_raphael_tf-explain_2021], shap and captum [@kokhlikyan_captum_2020].

\bibliography
