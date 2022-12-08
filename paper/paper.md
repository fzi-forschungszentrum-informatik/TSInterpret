---
title: 'TSInterpret: A Python package for the interpretability of time series classification'
tags:
  - Python
authors:
  - name: Jacqueline Höllig 
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Cedric Kulbach
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Steffen Thoma
    equal-contrib: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: FZI Forschungszentrum Informatik, Germany
   index: 1
date: 13 August 2017
bibliography: paper.bib

---

# Summary

`TSInterpret` is a python package that enables the post-hoc interpretability and explanation of black-box time series classifiers with three lines of code. Due to the specific structure of time serie (- i.e., non-independent features `[@ismail_benchmarking_2020]`, unintuitve visualizations `[@siddiqui_tsviz_2019]`), traditional interpretability and explainability libraries (e.g., `[@kokhlikyan_captum_2020;@klaise_alibi_2021;@meudec_raphael_tf-explain_2021]` ) find limit usage and do not not address time-series. `TSInterpret` specifically addresses the issue of black-box time-serie classification by providing a unified interface to state-of-the-art interpretation algorithms in combination with default plots. In addition the package provides a framework for developing additional easy-to-use interpretability methods.

# Statement of need

Temporal data is ubiquitous and encountered in many real-world applications ranging from electronic health records `[@rajkomar_scalable_2018]` to cyber security `[@susto_time-series_2018]`. Although deep learning methods have been successful in the field of Computer Vision (CV) and Natural Language Processing (NLP) for almost a decade, application on time series has only occurred in the past few years (e.g.,`[@fawaz_deep_2019;@rajkomar_scalable_2018;@susto_time-series_2018;@ruiz_great_2021]`. Deep learning models have been shown to achieve state-of-art results on time series classification (e.g., `[@fawaz_deep_2019]`). However, those methods are black-boxes due to their complexity which limits their application to high-stake scenarios (e.g., in medicine or autonomous driving), where user trust and understandability of the decision process are crucial. Much work has been done on interpretability in CV and NLP, most developed approaches are not directly applicable to time series data. The time component impedes the usage of existing methods `[@ismail_benchmarking_2020]`. Thus, increasing effort is put into adapting existing methods to time series (e.g., LEFTIST based on SHAP / Lime `[@guilleme_agnostic_2019]`, Temporal Saliency Rescaling for Saliency Methods `[@ismail_benchmarking_2020]`, Counterfactuals `[@ates_counterfactual_2021;sanchez-ruiz_instance-based_2021]`), and developing new methods specifically for time series interpretability (e.g., TSInsight based on autoencoders (Siddiqui et al. (2021)), TSViz for interpreting CNN (Siddiqui et al. (2019))). Compared to images or textual data, humans cannot intuitively and instinctively understand the underlying information contained in time series data. Therefore, time series data, both uni- and multivariate, have an unintuitive nature, lacking an understanding at first sight  `[@siddiqui_tsviz_2019]`. Hence, providing suitable isualizations of time series interpretability becomes crucial.

# Features

Explanations can take on various form (see Figure  \autoref{fig:Example}). Different use cases or users need different types of explanations. While for a domain expert counterfactuals might be useful, a data scientist or machine learning engineet prefere gradient based approaches `[@ismail_benchmarking_2020]` to evaluate the models feature attribution.
![Explanations.\label{fig:Example}](ECG.png)

Counterfactual approches calculate counter examples by finding a time series close to the original time series that is classified differently, thereby showing decision boundries. The intuition is to answer the question 'What if ?'. `TSInterpret` implements `@ates_counterfactual_2021` a perturbation based approach for multivariate data and `@sanchez-ruiz_instance-based_2021` for univartiate time series.
Gradient-based approaches (e.g., GradCam ) were adapted to time series by `[@ismail_benchmarking_2020]` that proposed rescaling according to time step importande and feature importance. It is applicaple to both gradient and perturbation based methods and based on tf-explain and captum. 
LEFTIST by `[@guilleme_agnostic_2019]` calculates feature importance based on a variety of Lime based on shapelets.

![Architecture of TSInterpret.\label{fig:Architecture}](Taxonomy.png){ width=20% }

`TSInterpret` implements these algorithms, according to the taxonomy shown in \autoref{fig:Architecture}. The interpretability methods are sorted according to a) accoring to the model output (e.g., is a feature map returned or an example time series) and b) the used mechanism (e.g., XXX) . Thereby, all implemented objects share a consistent interface to ensure that all methods contain a method explain and a plot function. The plot function is implemented on the level below based on the output structure provided by the interpretability algorithm to provide a unified visualization experience (e.g., in the case of Feature Attribution, the plot function visualizes a heatmap on the original sample). If necessary, those plots are refined by the Mechanism layer. The explain function is implemented on the method level.
This ensures the consistency extensiability of the framework.
XXX TODO High reuseability 


# Acknowledgements

This work was carried out with the support of the German Federal Ministry of Education
and Research (BMBF) within the project ”MetaLearn” (Grant 02P20A013).

# References