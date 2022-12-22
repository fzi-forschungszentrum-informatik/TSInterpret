# Explainable and Interpretable Machine Learning

Explainable Artificial Intelligence (XAI) is an emerging field trying to make AI sytems more understandable to humans. The goal of XAI according to DARPA[@gunning2017explainable], is to “produce more explainable models, while maintaining a high level of learning performance (prediction accuracy); and enable human users to understand, appropriately, trust, and effectively manage the emerging generation of artificially intelligent partners”.  Especially, on deep networks in high-stake sceanrios understanding the decison process or the decision is crucial to prevent harm (e.g., autonmous driving or anything health related).

Multiple terms are strongly related to XAI. Most famously "Explainability" and "Interpretability". Both terms are often used interchangably with no consent on definitions existing in literature. 

"Interpretability" in the context of TSInterpret refers to the ability to support user understanding and comprehension of the model decision making process and predictions. Used to provide user undestanding of model decison "Explainability" algorithms are used. Thereby, it is is often the case that multiple explainability algorithms are necessary for the user to understand the decision process.

"Explainability" tries provide algorithms that give insights into model predictions. 

- How does a prediction change dependent on feature inputs?

- What features are or are not important for a given prediction to hold?

- What set of features would you have to minimally change to obtain a new prediction of your choosing?

- How does each feature contribute to a model’s prediction?

Interpretability is the end-goal, explanations and explainability are tools to reach interpretability [@honegger2018shedding].

TSInterpret provides a set of algorithms or methods known as explainers specifccaly for time series. Each explainer provides different kind of insight about a model (- i.e., answers different types of questions). The set of algorithms available to a specific model is dependent on a number of factors. For instance, some approaches need a gradient to funtion and can therefore only be applied to models providing such. A full listing can be found in the section Algorithm Overview. 


## Application
As machine learning methods have become more complex and more mainstream, with many industries now incorporating AI in some form or another, the need to understand the decisions made by models is only increasing. Explainability has several applications of importance.

Trust: At a core level, explainability builds trust in the machine learning systems we use. It allows us to justify their use in many contexts where an understanding of the basis of the decision is paramount. This is a common issue within machine learning in medicine, where acting on a model prediction may require expensive or risky procedures to be carried out.

Testing: Explainability might be used to audit financial models that aid decisions about whether to grant customer loans. By computing the attribution of each feature towards the prediction the model makes, organisations can check that they are consistent with human decision-making. Similarly, explainability applied to a model trained on image data can explicitly show the model’s focus when making decisions, aiding debugging. Practitioners must be wary of misuse, however.

Functionality: Insights can be used to augment model functionality. For instance, providing information on top of model predictions such as how to change model inputs to obtain desired outputs.

Research: Explainability allows researchers to understand how and why models make decisions. This can help them understand more broadly the effects of the particular model or training schema they’re using.



## Taxonomy 
Explanations Methods and Techneiques for Model Interpretability can be classified according to different criterias. In this section we only introduce the criterias most relevant to TSInterpret.

### Post-Hoc vs Instrinct

Instrinct Interpretability refers to models that are interpretable by design. This can be achieved by constraing model complexity or inclusion of explanation components into the model design. 

Post-Hoc Interpretability refers to explanation methods applied after model training and are usually decoupled from the model. 

TSInterpret focuses on Post-Hoc Interpretability.

### Model-Specific vs Model-Agnostic

Model-Specific methods are limited to specific model classes and usually rely on specific model internal (e.g., Gradients).
Model-Agnostic methods can be applied to any model and rely on analyzing the connection between inputs and output. Those mehtods cannot access the model internal functions.

### Results of Explanation Methods

- TODO 
- TODO
- TODO

#TODO Input Architecture

## Simple Example 
<img src="img/Post-Hoc.png" height=300 width=300 />
Take for example a Decision Support System to classify heart rates as depicted in the figure below. While the data scientist knows that the machine learning model is able to obtain an accuracy of over 90 % to classify a heart rate as abnormal or normal, the decision process of such a system is still intransprent resulting in unsureness about the decision process of a model.  To make this decision process more opaque a data scientist might decide to use algorithms for explainable and interpretable machine learning, to learn a) which features are important, b) which feature influence the decision of a model in a postive or negativ way?, c) how a counter example would look like ?. 



#TODO Input from Mails 

