import numpy as np


class HenonMap:
    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b

    def model(self, x: float, y: float):
        return (self._a - x * x + self._b * y, x)

    def simulate(self, x_init: float, y_init: float, steps: int):
        XY: np.ndarray = np.array([[x_init, y_init]])
        for _ in range(steps):
            x, y = self.model(XY[-1][0], XY[-1][1])
            XY = np.vstack([XY, [x, y]])

        return XY

    def fixed_points(self):
        tmp1 = (self._b - 1) * np.sqrt(
            self._b * self._b - 2 * self._b + 1 + 4 * self._a
        )
        tmp2 = self._b * self._b - 2 * self._b + 1 + 4 * self._a
        tmp3 = 2 * np.sqrt(self._b * self._b - 2 * self._b + 1 + 4 * self._a)
        return np.array(
            [
                [(tmp1 + tmp2) / tmp3, (tmp1 + tmp2) / tmp3],
                [(tmp1 - tmp2) / tmp3, (tmp1 - tmp2) / tmp3],
            ]
        )

    def jacob(self, x: float, y: float):
        return np.array([[-2 * x, self._b], [1, 0]])
