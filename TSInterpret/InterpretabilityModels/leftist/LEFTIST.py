from TSInterpret.InterpretabilityModels.leftist.learning_process.learning_process import LearningProcess
from TSInterpret.InterpretabilityModels.leftist.learning_process.utils_learning_process import predict_proba
from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase
from TSInterpret.InterpretabilityModels.FeatureAttribution import FeatureAttribution
from TSInterpret.InterpretabilityModels.leftist.timeseries.transform_function.mean_transform import MeanTransform
from TSInterpret.InterpretabilityModels.leftist.timeseries.transform_function.rand_background_transform import RandBackgroundTransform
from TSInterpret.InterpretabilityModels.leftist.timeseries.transform_function.straightline_transform import StraightlineTransform
from TSInterpret.InterpretabilityModels.leftist.learning_process.SHAP_learning_process import SHAPLearningProcess
from TSInterpret.InterpretabilityModels.leftist.learning_process.LIME_learning_process import LIMELearningProcess
from TSInterpret.InterpretabilityModels.leftist.timeseries.segmentator.uniform_segmentator import UniformSegmentator
import matplotlib.pyplot as plt 
from TSInterpret.Models.PyTorchModel import PyTorchModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel
from TSInterpret.Models.SklearnModel import SklearnModel


class LEFTIST(FeatureAttribution):
    """
    Local explainer for time series classification.

    Attributes:
        transform (python function): the function to generate neighbors representation from interpretable features.
        segmenetator (python function): the function to get the interpretable features.
        model_to_explain (python function): the model to explain, must returned proba as prediction.
        learning_process (LearningProcess): the method to learn the explanation model.
    TODO This section is still to do
    """
    def __init__(self,model_to_explain, reference_set = None,mode='time',backend='F'):
        '''
         Args:
            model_to_explain: classification model to explain
            reference_set: reference set
            transform_name: name of transformer to be used
            segmentator: name of segmenator to be used
            learning_process_name: Either Lime or Shap 
            mode: time or Feature
            backend: TF, PYT or SK 
        '''
        super().__init__(model_to_explain, mode)

        self.neighbors = None
        # TODO move transform, segmentor and, learning process to EXPLAIN 
 
        self.test_x=reference_set
        self.backend=backend
        self.mode = mode

        if backend == 'PYT':
            self.predict=PyTorchModel(self.model, mode=mode).predict
            
        elif backend== 'TF':
            self.predict=TensorFlowModel(self.model,mode='time').predict
            #Parse test data into torch format : 
            
        elif backend=='SK': 
            self.predict=SklearnModel(self.model,mode='time').predict
        else:
            #Assumption this is already a predict Function 
            print('The Predict Function was given directly')
            self.predict=self.model


    def explain(self,instance,nb_neighbors, idx_label=None, explanation_size=None,transform_name='straight', segmentator_name='uniform', learning_process_name='Lime',nb_interpretable_feature=10, random_state=0):
        """
        Compute the explanation model.

        Parameters:
            nb_neighbors (int): number of neighbors to generate.
            explained_instance (np.ndarray): time series instance to explain
            idx_label (int): index of label to explain. If None, return an explanation for each label.
            explanation_size (int): number of feature to use for the explanations

        Returns:
            An explanation model for the desired label

        TODO
            * Careful changed siganture

        """
        print('Instance', instance.shape)
        self.transform = transform_name
        self.transform_name=transform_name
        self.learning_process_name = learning_process_name
        if segmentator_name=='uniform':
            self.segmentator = UniformSegmentator(nb_interpretable_feature)

        if self.mode =='feat':
            instance=instance.reshape(instance.shape[-1],instance.shape[-2])
        if self.transform_name == 'mean':
            self.transform = MeanTransform(instance)
        elif self.transform_name == 'straight_line':
            self.transform = StraightlineTransform(instance)
        else:
            self.transform = RandBackgroundTransform(instance)
            self.transform.set_background_dataset(self.test_x)
        if self.learning_process_name == 'SHAP':
            self.learning_process = SHAPLearningProcess(instance,self.model_to_explain,external_dataset=self.test_x)
        else:
            self.learning_process = LIMELearningProcess(random_state)
        # get the number of features of the simplified representation
        nb_interpretable_features, segments_interval = self.segmentator.segment(instance)

        self.transform.segments_interval = segments_interval

        # generate the neighbors around the instance to explain
        self.neighbors = self.learning_process.neighbors_generator.generate(nb_interpretable_features, nb_neighbors,self.transform)

        # classify the neighbors
        self.neighbors = predict_proba(self.neighbors, self.model,self.backend,self.mode)

        # build the explanation from the neighbors
        if idx_label is None:
            explanations = []
            for label in range(self.neighbors.proba_labels.shape[1]):
                explanations.append(self.learning_process.solve(self.neighbors, label, explanation_size=explanation_size))
        else:
            explanations = [self.learning_process.solve(self.neighbors, idx_label, explanation_size=explanation_size)]
        
        #TODO reform the explanation ! 
        #values_per_slice=len(instance.reshape(-1))/10
        #exp= instance.copy()
        #for a in range(0,nb_interpretable_feature):

        return explanations
    
    def plot_on_sample(self,series,exp):
        '''TODO Visualizations
                * Include average of the counter class?
                * does this make sense ?  
                * save oprions for plot
        '''
        values_per_slice=len(series)/len(exp[0][0])
        step=0
        plt.Figure()
        plt.plot(series)
        print(exp[0][0])
        for i in range(0,len(exp[0][0])):
            weight=exp[0][0][i]
            print(weight)
            start = i * values_per_slice
            print(start)
            end = start + values_per_slice
            color = 'red' if weight < 0 else 'green'
            plt.axvspan(start, end, color=color, alpha=abs(weight * 2))

        plt.show()
    def plot():
        pass

