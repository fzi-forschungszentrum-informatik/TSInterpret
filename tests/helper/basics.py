import unittest
import random, torch, tensorflow
import numpy as np 


def set_all_random_seeds(seed: int = 1234) -> None:
    #TODO tensorflow and sklearn is missing
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all tests by providing a set up fixture,
    which sets a fixed random seed. Since many initializations are random, 
    this ensures that tests run deterministically.
    """

    def setUp(self) -> None:
        set_all_random_seeds(1234)
        #TODO What does this do ? 
        #patch_methods(self)