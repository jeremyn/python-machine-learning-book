import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import comb


def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [
        comb(n_classifier, k) * error**k * (1-error)**(n_classifier-k)
        for k in range(k_start, n_classifier+1)
    ]
    return sum(probs)


def plot_ensemble_error():
    error_range = np.arange(0.0, 1.01, 0.01)
    ensemble_errors = [
        ensemble_error(n_classifier=11, error=error) for error in error_range
    ]

    plt.plot(
        error_range,
        ensemble_errors,
        label='Ensemble error',
        linewidth=2,
    )
    plt.plot(
        error_range,
        error_range,
        linestyle='--',
        label='Base error',
        linewidth=2,
    )
    plt.xlabel('Base error')
    plt.ylabel('Base/Ensemble error')
    plt.legend(loc='upper left')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    plot_ensemble_error()
