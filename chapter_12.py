import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


NN__MNIST_FILENAME = 'nn_mnist.pkl'


def get_mnist_data():
    path = os.path.join('datasets', 'mnist')
    mnist_data = []
    for kind in ('train', 't10k'):
        labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
        images_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)

        with open(labels_path, 'rb') as lbpath:
            lbpath.seek(8)
            mnist_data.append(np.fromfile(lbpath, dtype=np.uint8))

        with open(images_path, 'rb') as imgpath:
            imgpath.seek(16)
            mnist_data.append(
                np.fromfile(
                    imgpath,
                    dtype=np.uint8,
                ).reshape(len(mnist_data[-1]), 784)
            )

    y_train, X_train, y_test, X_test = mnist_data
    print(
        "Train: rows: %d, columns: %d" % (X_train.shape[0], X_train.shape[1])
    )
    print("Test: rows: %d, columns: %d" % (X_test.shape[0], X_test.shape[1]))
    return X_train, X_test, y_train, y_test


def display_mnist_examples(X, y):
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = X[y == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()


class NeuralNetMLP(object):
    """Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ----------
    n_output : int
      Number of output units, should be equal to the number of unique class
      labels.

    n_features : int
      Number of features (dimensions) in the target dataset. Should be equal
      to the number of columns in the X array.

    n_hidden : int (default: 30)
      Number of hidden units.

    l1 : float (default: 0.0)
      Lambda value for L1-regularization. No regularization if l1=0.0 (default)

    l2 : float (default: 0.0)
      Lambda value for L2-regularization. No regularization if l2=0.0 (default)

    epochs : int (default: 500)
      Number of passes over the training set.

    eta : float (default: 0.001)
      Learning rate.

    alpha : float (default: 0.0)
      Momentum constant. Factor multiplied with the gradient of the previous
      epoch t-1 to improve learning speed.

    decrease_const : float (default: 0.0)
      Decrease constant. Shrinks the learning rate after each epoch via
      eta / (1 + epoch*decrease_const)

    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent circles.

    minibatches : int (default: 1)
      Divides training data into k minibatches for efficiency. Normal gradient
      descent learning if k=1 (default).

    random_state : int (default: None)
      Set random state for shuffling and initializing the weights.

    Attributes
    ----------
    cost_: list
      Sum of squared errors after each epoch.

    """
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0,
                 epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0,
                 shuffle=True, minibatches=1, random_state=None, debug=False):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.debug = debug

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ----------
        y : array, shape = (n_samples, )
            Target values.

        Returns
        -------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for index, val in enumerate(y):
            onehot[val, index] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(
            -1.0,
            1.0,
            size=self.n_hidden*(self.n_features+1),
        )
        w1 = w1.reshape(self.n_hidden, self.n_features+1)
        w2 = np.random.uniform(
            -1.0,
            1.0,
            size=self.n_output*(self.n_hidden+1),
        )
        w2 = w2.reshape(self.n_output, self.n_hidden+1)
        return w1, w2

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow error for very small input
        values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient function of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1]+1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError("'how' must be 'column' or 'row"'')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Compute feedforward step

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
          Input layer with original features.

        w1 : array, shape = (n_hidden_units, n_features)
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = (n_output_units, n_hidden_units)
          Weight matrix for hidden layer -> output layer.

        Returns
        -------
        a1 : array, shape = (n_samples, n_features+1)
          Input values with bias unit.

        z2 : array, shape = (n_hidden, n_samples)
          Net input of hidden layer.

        a2 : array, shape = (n_hidden+1, n_samples)
          Activation of hidden layer.

        z3 : array, shape = (n_output_units, n_samples)
          Net input of output layer.

        a3 : array, shape = (n_output_units, n_samples)
          Activation of output layer.

        """
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2-regularization cost"""
        return (lambda_/2.0) * (np.sum(w1[:, 1:]**2) + np.sum(w2[:, 1:]**2))

    def _L1_reg(self, lambda_, w1, w2):
        """Compute L1-regularization cost"""
        return (
            (lambda_/2.0) *
            (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())
        )

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.

        y_enc : array, shape = (n_labels, n_samples)
          One-hot encoded class labels.

        output : array, shape = (n_output_units, n_samples)
          Activation of the output layer (feedforward)

        w1 : array, shape = (n_hidden_units, n_features)
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = (n_output_units, n_hidden_units)
          Weight matrix for hidden layer -> output layer.

        Returns
        -------
        cost : float
          Regularizaed cost.

        """
        term1 = -y_enc * np.log(output)
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """Compute gradient step using backpropagation.

        Parameters
        ----------
        a1 : array, shape = (n_samples, n_features+1)
          Input values with bias unit.

        a2 : array, shape = (n_hidden+1, n_samples)
          Activation of hidden layer.

        a3 : array, shape = (n_output_units, n_samples)
          Activation of output layer.

        z2 : array, shape = (n_hidden, n_samples)
          Net input of hidden layer.

        y_enc : array, shape = (n_labels, n_samples)
          One-hot encoded class labels.

        w1 : array, shape = (n_hidden_units, n_features)
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = (n_output_units, n_hidden_units)
          Weight matrix for hidden layer -> output layer.

        Returns
        -------
        grad1 : array, shape = (n_hidden_units, n_features)
          Gradient of the weight matrix w1.

        grad2 : array, shape = (n_output_units, n_hidden_units)
          Gradient of the weight matrix w2.

        """
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += w1[:, 1:] * (self.l1 + self.l2)
        grad2[:, 1:] += w2[:, 1:] * (self.l1 + self.l2)

        return grad1, grad2

    def _gradient_checking(self, X, y_enc, w1, w2, epsilon, grad1, grad2):
        """Apply gradient checking (for debugging only)

        Returns
        -------
        relative_error : float
          Relative error between the numerically approximated gradients and
          the backpropagated gradients.

        """
        num_grad1 = np.zeros(np.shape(w1))
        epsilon_arr1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_arr1[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1-epsilon_arr1, w2)
                cost1 = self._get_cost(y_enc, a3, w1-epsilon_arr1, w2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1+epsilon_arr1, w2)
                cost2 = self._get_cost(y_enc, a3, w1+epsilon_arr1, w2)
                num_grad1[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_arr1[i, j] = 0

        num_grad2 = np.zeros(np.shape(w2))
        epsilon_arr2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_arr2[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2-epsilon_arr2)
                cost1 = self._get_cost(y_enc, a3, w1, w2-epsilon_arr2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2+epsilon_arr2)
                cost2 = self._get_cost(y_enc, a3, w1, w2+epsilon_arr2)
                num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_arr2[i, j] = 0

        num_grad = np.hstack((num_grad1.flatten(), num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error = norm1 / (norm2 + norm3)
        return relative_error

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
          Input layer with original features.

        Returns
        -------
        y_pred : array, shape = (n_samples, )
          Predicted class labels.

        """
        if len(X.shape) != 2:
            raise AttributeError(
                "X must be an (n_samples, n_features) array. Use X[:, None] "
                "for 1-feature classification, or X[[i]] for 1-sample "
                "classification"
            )

        a1, z2, s2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """Learn weights from training data.

        Parameters
        ----------
        X : array, shape = (n_samples, n_faetures)
          Input layer with original features.

        y : array, shape = (n_samples, )
          Target class labels.

        print_progress : bool (default: False)
          Prints progress as the number of epochs to stderr.

        Returns
        -------
        self

        """
        self.cost_ = []
        X_data = X.copy()
        y_data = y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= 1 + self.decrease_const*i

            if print_progress:
                sys.stderr.write("\rEpoch: %d/%d" % (i+1, self.epochs))

            if self.shuffle:
                index = np.random.permutation(y_data.shape[0])
                X_data = X_data[index]
                y_enc = y_enc[:, index]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for index in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(
                    X_data[index],
                    self.w1,
                    self.w2,
                )
                cost = self._get_cost(
                    y_enc=y_enc[:, index],
                    output=a3,
                    w1=self.w1,
                    w2=self.w2,
                )
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(
                    a1=a1,
                    a2=a2,
                    a3=a3,
                    z2=z2,
                    y_enc=y_enc[:, index],
                    w1=self.w1,
                    w2=self.w2,
                )

                if self.debug:
                    grad_diff = self._gradient_checking(
                        X=X_data[index],
                        y_enc=y_enc[:, index],
                        w1=self.w1,
                        w2=self.w2,
                        epsilon=1e-5,
                        grad1=grad1,
                        grad2=grad2,
                    )

                    if grad_diff <= 1e-7:
                        status = "Ok"
                    elif grad_diff <= 1e-4:
                        status = "Warning"
                    else:
                        status = "PROBLEM"
                    print("%s: %s" % (status, grad_diff))

                delta_w1 = self.eta * grad1
                delta_w2 = self.eta * grad2
                self.w1 -= delta_w1 + (self.alpha * delta_w1_prev)
                self.w2 -= delta_w2 + (self.alpha * delta_w2_prev)
                delta_w1_prev = delta_w1
                delta_w2_prev = delta_w2

        return self


def get_trained_nn(X_train, y_train):
    np.random.seed(1)
    try:
        nn = pickle.load(open(NN__MNIST_FILENAME, 'rb'))
    except FileNotFoundError:
        nn = NeuralNetMLP(
            n_output=10,
            n_features=X_train.shape[1],
            n_hidden=50,
            l2=0.1,
            l1=0.0,
            epochs=1000,
            eta=0.001,
            alpha=0.001,
            decrease_const=0.00001,
            minibatches=50,
            shuffle=False,
            random_state=1,
        )
        nn.fit(X_train, y_train, print_progress=True)
        pickle.dump(nn, open(NN__MNIST_FILENAME, 'wb'), protocol=4)

    return nn


def display_nn_performance(nn):
    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.show()

    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_arr = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_arr[i]) for i in batches]

    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.show()

    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print("Training accuracy: %.2f%%" % (acc * 100))

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print("Testing accuracy: %.2f%%" % (acc * 100))

    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title(
            "%d) t: %d p: %d" %
            (i+1, correct_lab[i], miscl_lab[i])
        )

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()


def run_nn_check(X, y):
    nn_check = NeuralNetMLP(
        n_output=10,
        n_features=X.shape[1],
        n_hidden=10,
        l2=0.0,
        l1=0.0,
        epochs=10,
        eta=0.001,
        alpha=0.0,
        decrease_const=0.0,
        minibatches=1,
        shuffle=False,
        random_state=1,
        debug=True,
    )
    nn_check.fit(X[:5], y[:5], print_progress=False)


if __name__ == '__main__':
    path = os.path.join('datasets', 'mnist')
    X_train, X_test, y_train, y_test = get_mnist_data()
    # display_mnist_examples(X_train, y_train)
    # nn = get_trained_nn(X_train, y_train)
    # display_nn_performance(nn)
    run_nn_check(X_train, y_train)
