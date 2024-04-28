import numpy as np


class DiscreteLogisticGrowth:
    def __init__(self, a: float):
        self._a = a

    def model(self, x: float | np.ndarray):
        return self._a * x * (1 - x)

    def simulate(self, inits: float | list[float], steps: int):
        if isinstance(inits, float):
            inits = [inits]

        X = np.array([[x] for x in inits])

        for _ in range(steps):
            x = self.model(X[:, -1:])
            X = np.concatenate((X, x), axis=1)
        return X

    def web_simulate(
        self,
        init: float,
        steps: int,
        web_y_init: float = 0.0,
        x_range: np.ndarray = np.linspace(0, 1, 100),
    ):
        Web_x = np.array([init])
        Web_y = np.array([web_y_init])

        for _ in range(steps):
            x = self.model(Web_x[-1])
            Web_x = np.append(Web_x, [Web_x[-1], x])
            Web_y = np.append(Web_y, [x, x])

        y_model_map = self.model(x_range)

        return Web_x, Web_y, y_model_map, x_range

    def fixed_points(self):
        return list(set([0.0, (self._a - 1) / self._a]))
