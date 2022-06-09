from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import math
import numpy as np

#TODO from leftist / Experiments 
#TODO Pointing Game / Evlauation Testing 

def faithfulness(x_train,y_train,x_test,y_test, explainer, max_number_features):
    '''
    Implemented after Ates et al. 2020
    Train Logisic Regression with  L1
    Change L1 until less than 10 feaures are used
    • Recall: How many of the metrics used by the classifier are in the explanation?
    • Precision: How many of the metrics in the explanation are used by the classifier?

    Set Parameters, so that all have the same amount of features.
    TODO
        * Check Implementation
        * Currently only works for Lime
        * Other Implementation ? Decison Tree etc ?
        * How to cope with slicing ?
        * Max Number features
    '''
    lr=LogisticRegression(penalty='l1',solver='liblinear')
    lr.fit(x_train.reshape(-1, x_train.shape[1]),y_train)
    explanation=[]
    values_per_slice = math.ceil(x_train.shape[0] / 17)
    for a in x_test:
        #Get Explanations for all Training Data
        local_explanation= explainer(np.array(a),lr.predict_proba,num_features=max_number_features,num_slices=17)#len(a))
        ex=[]
        for i in range(max_number_features):
            feature, weight = local_explanation.as_list()[i]
            start = i * values_per_slice
            print(start)
            end = start + values_per_slice
            #used to be feature
            if weight!= 0:
                ex.append(start)

        explanation.append(ex)
    print(explanation)
    _,weights =np.where(lr.coef_!=0)
    print(weights)
    length_weights_non_zero= len(weights)
    identical = 0
    for weight in weights:
        for a in explanation:
            print(int(a))
            #length_item = len(a)
            if a in weight:
                print(weight)
                identical= identical+1
    if identical>0:
        recall_overall =identical/(length_weights_non_zero*len(x_test))
        precision_overall = (length_weights_non_zero * len(x_test)) / identical
    else:
        recall_overall=0
        precision_overall = 0
    print('Precision', precision_overall)
    print('Recall',recall_overall)

def _explanation_to_numpy(item,explanation,use_binary=False):
    ex=np.zeros_like(item)
    for i, value in explanation:
        #print(i)
        #print(value)
        if use_binary:
            ex[i] = 1
        else:
            ex[i]=value
    return ex


def roboustness(x_test,explainer,model):
    '''
    Implemented after Ates et al. 2020
    Lipshitz constant  for each test sample, average the results
    TODO :
        *make more efficient by using indices
        *Calculate all explanations as one ?
        *Is this calculation correct
    '''
    #TODO Flexible
    #Clacculate k- Nearest Neighbor (Currently testing it with 5 )
    knn=NearestNeighbors(n_neighbors=5)
    knn.fit(x_test.reshape(-1,286))
    all_lip=[]
    for item in x_test:
        tested_item = explainer(np.array(item), model, num_features=7, num_slices=17).local_exp[1]
        tested_item=_explanation_to_numpy(item,tested_item,True)
        #local_exp
        neighbors= knn.kneighbors(item.reshape(1,-1),n_neighbors=5,return_distance=False)
        #print(neighbors)
        #Calculate Explantion for neigbors
        explanation=[]
        samples=[]
        for i in neighbors[0]:
            saver=explainer(np.array(x_test[i].reshape(286,1)), model, num_features=7, num_slices=17).local_exp[1]
            explanation.append(_explanation_to_numpy(item, saver,True))
            samples.append(x_test[i])
            #print('explanation ', i)
        #print(len(explanation))
        tested_item= np.repeat(tested_item,5 )
        item = np.repeat(item,5)
        lipshitz_index= np.max( np.linalg.norm(tested_item-explanation)/np.linalg.norm(item-samples))
        all_lip.append(lipshitz_index)
    print('Lipshitz_index', np.average(np.array(all_lip)))

    return None

def precision():
    '''
    Implemented after Ismail et al. 2020
    '''
    return None

def recall():
    '''
    Implemented after Ismail et al. 2020
    '''
    return None

def gernealizable():
    '''
    Implemented after Ates et al. 2020
    misclassified test instance, we get an explanation and apply the same metric substitutions using the same distractor to other test samples with the same (true class, predicted class) pair.
    percentage of misclassifications that the
    explanation applies to (i.e., successfully flips the prediction for)
    '''
    return None
def useful_for_Understanding():
    '''Implemented after Ates et al. 2020'''
    return None
def comprehensability():
    '''Implemented after Ates et al. 2020
    Multivariate --> number of metrics /features
    '''
    return None

def qualtitative_evaluation():
    #TODO not here extra class
    '''
    Implemented after Ates et al. 2020
    Frame work from Tuncer et al. [8], [9],
    '''
    #Random Forest Classifier witg feature extraction memleak anmly
    return None

def faithfullness_fast_shapelets():
    '''Implement after Agnostic Local Explanation for time-series classification'''
    return None