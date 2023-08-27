from agent import TwoLayerFC
import numpy as np


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not np.array_equal(p1.detach().numpy(), p2.detach().numpy()):
            return False
    return True


model1 = TwoLayerFC(12, 2, 64)
model2 = TwoLayerFC(12, 2, 64)
print(compare_models(model1, model2))