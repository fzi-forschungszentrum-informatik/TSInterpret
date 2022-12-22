import numpy as np

from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
from TSInterpret.InterpretabilityModels.counterfactual.TSEvo.Evo import (
    EvolutionaryOptimization,
)


class TSEvo(CF):
    """
    Calculates and Visualizes Counterfactuals for Uni- and Multivariate Time Series in accordance to the paper [1].

    References
    ----------
     [1] Höllig, Jacqueline , et al.
     "TSEvo: Evolutionary Counterfactual Explanations for Time Series Classification."
     21st IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2022.
    ----------
    """

    def __init__(self, model, data, mode="time", backend="PYT", verbose=0):
        """
        Arguments:
            model [torch.nn.Module, Callable, tf.keras.model]: Model to be interpreted.
            data Tuple: Reference Dataset as Tuple (x,y).
            mode str: Name of second dimension: time -> (-1, time, feature) or feat -> (-1, feature, time)
            backend str: desired Model Backend ('PYT', 'TF', 'SK').
            verbose int: Logging Level
        """
        super().__init__(model, mode)
        self.backend = backend
        self.verbose = verbose
        if type(data) == tuple:
            self.x, self.y = data
            # print('Len Reference Set ', len(self.x.shape))
            print(type(self.y[0]))
            if not type(self.y[0]) == int and not type(self.y[0]) == np.int64:
                print("y was one Hot Encoded")
                self.y = np.argmax(self.y, axis=1)
            if len(self.x.shape) == 2:
                print("Reshape Reference Set")
                self.x.reshape(-1, 1, self.x.shape[-1])
            # print('Reference Set Constructor',self.x.shape)
        else:
            self.x, self.y = None, None
            print("Dataset is no Tuple ")
        pass

    def explain(
        self,
        original_x,
        original_y,
        target_y=None,
        transformer="authentic_opposing_information",
        epochs=500,
    ):
        """
        Entry Point to explain a instance.
        Arguments:
            original_x (np.array): The instance to explain. Shape `mode = time` -> `(1,time, feat)` or `mode = time` -> `(1,feat, time)`
            original_y (np.array): Classification Probability of instance.
            target_y int: Class to be targeted
            transformer str: ['authentic_opposing','mutate_both','gaussian','frequency']
            epochs int: # Iterations
        Returns:
            [np.array, int]: Returns the Counterfactual and the class. Shape of CF : `mode = time` -> `(time, feat)` or `mode = time` -> `(feat, time)`
        """
        print(len(original_y.shape))
        if len(original_x.shape) < 3:
            original_x = np.array([original_x])
        if self.backend == "TF" or self.mode == "time":
            original_x = original_x.reshape(
                original_x.shape[0], original_x.shape[2], original_x.shape[1]
            )
        neighborhood = []
        if target_y is not None:
            if not type(target_y) == int:
                target_y = np.argmax(original_y, axis=1)[0]
            reference_set = self.x[np.where(self.y == target_y)]
        elif len(original_y) > 1:
            reference_set = self.x[np.where(self.y != np.argmax(original_y, axis=1)[0])]
        else:
            reference_set = self.x[np.where(self.y != original_y)]

        reference_set = reference_set.reshape(
            -1, original_x.shape[1], original_x.shape[2]
        )
        if len(reference_set.shape) == 2:
            reference_set = reference_set.reshape(-1, 1, reference_set.shape[-1])

        window = original_x.shape[-1]
        channels = original_x.shape[-2]
        e = EvolutionaryOptimization(
            self.model,
            original_x,
            original_y,
            target_y,
            reference_set,
            neighborhood,
            window,
            channels,
            self.backend,
            transformer,
            verbose=self.verbose,
            epochs=epochs,
        )
        ep, output = e.run()
        return np.array(ep)[0][0], output
