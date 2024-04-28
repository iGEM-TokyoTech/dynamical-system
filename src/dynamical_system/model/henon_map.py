import numpy as np


class HenonMap:
    def __init__(self, a: np.longdouble, b: np.longdouble):
        self._a = a
        self._b = b

    def model(self, x: np.longdouble, y: np.longdouble):
        return (self._a - x * x + self._b * y, x)

    def model_inverse(self, x: np.longdouble, y: np.longdouble):
        return (y, (x + y * y - self._a) / self._b)

    def simulate(self, x_init: np.longdouble, y_init: np.longdouble, steps: int):
        XY: np.ndarray = np.array([[x_init, y_init]], dtype=np.longdouble)
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
            ],
            dtype=np.longdouble,
        )

    def jacob(self, x: np.longdouble, y: np.longdouble):
        return np.array([[-2 * x, self._b], [1, 0]], dtype=np.longdouble)
