--
title: 'TSInterpret: A Python package for the interpretability of time series classification'
tags:
  - Python

authors:
  - name: Jacqueline Höllig
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Cedric Kulbach
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name:Steffen Thoma
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: FZI Forschungszentrum Informatik, Germany
   index: 1
date: 13 August 2017
bibliography: ./paper/paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`TSInterpret` is a python package that enables the post-hoc interpretability and explanation of black-box time series classifiers with three lines of code. It provides a unified interface to state-of-the-art interpretation algorithms in combination with default plots. In addition the package provides a framework for developing additional interpretability methods XXXX. Thereby `TSInterpret` supports classification models written in torch and tensorflow.

XXXX Build on 



# Statement of need
Temporal data is ubiquitous and encountered in many real-world applications ranging from electronic health records `[@rajkomar_scalable_2018]` to cyber security `[@susto_time-series_2018]`. Although almost omnipresent, time series classification has been considered one of the most challenging problems in data mining for the last two decades `[yang_10_2006;esling_time-series_2012]`. With the rising data availability and accessibility (e.g., provided by the UCR / UEA archive `[@bagnall_great_2017;dau_ucr_2019]`, hundreds of time series classification algorithms have been proposed. Although deep learning methods have been successful in the field of Computer Vision (CV) and Natural Language Processing (NLP) for almost a decade, application on time series has only occurred in the past few years (e.g.,`[@fawaz_deep_2019;@rajkomar_scalable_2018;@susto_time-series_2018;@ruiz_great_2021]`. Deep learning models have been shown to achieve state-of-art results on time series classification (e.g., `[@fawaz_deep_2019]`). However, those methods are black-boxes due to their complexity which limits their application to high-stake scenarios (e.g., in medicine or autonomous driving), where user trust and understandability of the decision process are crucial. Although much work has been done on interpretability in CV and NLP, most developed approaches are not directly applicable to time series data. The time component impedes the usage of existing methods `[@ismail_benchmarking_2020]`. Thus, increasing effort is put into adapting existing methods to time series (e.g., LEFTIST based on SHAP / Lime `[@guilleme_agnostic_2019]`, Temporal Saliency Rescaling for Saliency Methods `[@ismail_benchmarking_2020]`, Counterfactuals `[@ates_counterfactual_2021;sanchez-ruiz_instance-based_2021]`), and developing new methods specifically for time series interpretability (e.g., TSInsight based on autoencoders (Siddiqui et al. (2021)), TSViz for interpreting CNN (Siddiqui et al. (2019))). Compared to images or textual data, humans cannot intuitively and instinctively understand the underlying information contained in time series data. Therefore, time series data, both uni- and multivariate, have an unintuitive nature, lacking an understanding at first sight (Siddiqui et al. (2019)). Hence, providing suitable isualizations of time series interpretability becomes crucial.

# Features

Explanations can take on various form (see Figure XXX) shows XXX. For different use cases XXXX

# Example 


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

This work was carried out with the support of the German Federal Ministry of Education
and Research (BMBF) within the project ”MetaLearn” (Grant 02P20A013).

# References