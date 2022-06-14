from TSInterpret.InterpretabilityModels.InterpretabilityBase import InterpretabilityBase

class InstanceBase(InterpretabilityBase):
    def __init__(self, mlmodel, mode):
        super().__init__(mlmodel)
        self.mode = mode

    def explain(self):
        pass
    def plot (self):
        pass