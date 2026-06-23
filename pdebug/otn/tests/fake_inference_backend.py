from pdebug.otn.inference import InferenceRequest


class EchoBackend:
    def __init__(self, prefix="echo"):
        self.prefix = prefix

    def predict(self, request: InferenceRequest):
        return {
            "data": {
                "prefix": self.prefix,
                "inputs": request.inputs,
                "parameters": request.parameters,
            }
        }

    def close(self):
        self.closed = True
