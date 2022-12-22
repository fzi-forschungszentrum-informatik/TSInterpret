# Getting Started

TSInterpret works with Python 3.6+ and can be installed from PyPI or cloning the github repository by following the instructions below.

Via PyPI: 
```shell
pip install tsinterpret
```

You can install the latest development version from GitHub as so:
```shell
pip install https://github.com/fzi-forschungszentrum-informatik/TSInterpret.git --upgrade
```

Or, through SSH:
```shell
pip install git@github.com:fzi-forschungszentrum-informatik/TSInterpret.git --upgrade
```
Feel welcome to open an <a href= "https://github.com/fzi-forschungszentrum-informatik/TSInterpret/issues/new"> issue on GitHub </a> if you are having any trouble.

# Basic Usage 

TSInterpret takes inspiration from scikit-learn, consisting of distinct initialize, explain and plot steps. We will use the Nun-CF method on to illustrate the API.

First, we import the explainer:
``` py
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF
```

Next, we initialize it by passing it a model (or in this case also possible a predict function) and any other necessary arguments:
``` py
exp_model=NativeGuideCF(model,shape,(train_x,train_y), backend='PYT', mode='feat',method='dtw_bary_center')
```

Finally, we can call the explainer on a test instance which will return an Tuple or a list containing the explanation: 

``` py
exp,label=exp_model.explain(item, np.argmax(y_target,axis=1))
```

Afterwads we can use the Plot Function to obtain a visualiation of the explanation: 

``` py
exp_model.plot(item,np.argmax(y_target,axis=1)[0],exp,label)
```

The input and output details vary for each method. Therefore, getting familiar with the different [algorithms](AlgorithmOverview.md) makes sense.  