class NOT_ENOUGH_NEURON(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class INVALID_PARAMETER(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class MODEL_NOT_EQUIV(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)