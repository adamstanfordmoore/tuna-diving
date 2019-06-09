"""Module for conducting piecewise linear regressions, written

Author
------
R. Davis Born
August, 2018

Support Contact
---------------
R. Davis Born
rdborn1@gmail.com

Routines
--------
pos(x)
    Set negative values of input data to 0 and return the result.
I(x)
    Create an index vector which has value 1 where input data is nonzero.

Classes
-------
MixedLayerRegression
    A class for performing and storing piecewise linear regressions with a
    single breakpoint, for identifying the mixed layer depth in oceanographic
    data (i.e. the slope of the first segment is constrained to be 0)

References
----------
.. [1] Muggeo, V. M. (2003). Estimating regression models with unknown
   break-points. Statistics in medicine, 22(19), 3055-3071.

Notes
-----

"""

import numpy as np

def pos(x):
    """Set negative values of input data to 0 and return the result.

    Parameters
    ----------
    x : numpy.array
        Data whose negative values are to be set to 0

    Returns
    -------
    numpy.array
        Modified `x` with all negative values set to 0
    """

    x[x < 0.] = 0.
    return x

def I(x):
    """Create an index vector which has value 1 where input data is nonzero.

    Parameters
    ----------
    x : numpy.array
        Data to be passed to the indexing function

    Returns
    -------
    numpy.array
        Index vector which is 1 where `x` is nonzero and 0 otherwise
    """

    Ix = np.zeros_like(x)
    Ix[x] = 1
    return Ix

class MixedLayerRegression:
    """A class for performing and storing piecewise linear regressions with a
       single breakpoint where the slope of the first segment is constrained
       to be zero.

    Parameters
    ----------
    None

    Attributes
    ----------
    train_x : numpy.array
        Training data, dependent variable, used in the current fit
    train_y : numpy.array
        Training data, independent variable, used in the current fit
    alpha : float
        Slope of first segment of piecewise regression (constrained to be zero
        in this implementation)
    beta : float
        Difference in slope between first and second segment (since `alpha` is
        constrained to be zero, `beta` will be the slope of the second segment)
    gamma : float
        Difference (in the independent variable) between the segments at the
        breakpoint, `psi` (algorithm converges when `gamma` vanishes)
    psi : float
        Breakpoint location (in the dependent variable)
    intercept : float
        Value of independent variable when dependent variable is zero.

    Methods
    -------
    fit(x, y)
        Fit a piecewise linear regression with a single breakpoint to data.
    pred(x)
        Predict outputs for test inputs using a previously fit piecewise
        linear regression.
    __Us__(x)
        Additional variate for regression on second segment.
    __Vs__(x)
        Additional variate for regression on breakpoint location.
    __initialize_psi__(x)
        Randomly initialize breakpoint location based on training data.
    __find_psi__()
        Fit piecewise linear regression and find breakpoint per [1].

    Examples
    --------
    >>> # Import the necessary modules
    >>> import numpy as np
    >>> from piecewise_linear_regression import MixedLayerRegression
    >>>
    >>> # Initialize your MixedLayerRegression object
    >>> my_mlr = MizedLayerRegression()
    >>>
    >>> # Fit the model
    >>> # (x_train and y_train are whatever data you are fitting)
    >>> my_mlr.fit(x_train, y_train)
    >>>
    >>> # Get the results of the model over the span of the training data
    >>> x_test = np.linspace(min(x_train), max(x_train), 1000)
    >>> y_pred = my_mlr.pred()

    Notes
    -----
    There is currently no built-in functionality to check training and test
    statistics to evaluate the model fit, or to hold a portion of the training
    data out for testing/evaluation.

    The author of [1] may have written an R package implementing his methods
    that may have more functionality than this Python module, but I have
    not investigated that.

    """

    def __init__(self):
        """Initialize regression object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def __Us__(self, z):
        """Additional variate for regression on second segment.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return pos(z - self.psi)

    def __Vs__(self, z):
        """Additional variate for regression on breakpoint location.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return -np.array(I(z > self.psi), dtype=int)

    def __initialize_psi__(self):
        """Randomly initialize breakpoint location based on training data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        x = self.train_x
        psi = np.std(x) * np.random.randn(1) + np.mean(x)
        return psi

    def __find_psi__(self):
        """Fit piecewise linear regression and find breakpoint per [1].

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Get data to fit the model
        x = self.train_x
        y = self.train_y

        # Initialize model parameters
        psi = self.__initialize_psi__()
        alpha = 0. # slope of the first segment is constrained to be zero
        beta = 10.
        gamma = 10.
        psi_p = np.inf

        # Max iterations (arbitrary choise)
        i = 1000

        # Iterative fit, stopping condition on gamma
        while abs(gamma) > 1e-2:
            self.psi = psi
            # Fitting procedure from [1]
            u = self.__Us__(x)
            v = self.__Vs__(x)
            c = np.ones_like(x)
            A = np.vstack([u, v, c]).T
            beta, gamma, intercept = np.linalg.lstsq(A, y)[0]
            psi_p = psi
            psi = psi + gamma / beta

            # Re-initialize breakpoint (psi) if it leaves the data's domain
            if psi > np.max(x) or psi < np.min(x):
                psi = self.__initialize_psi__()

            # Decrement iterations
            i -= 1

            # Give up if stopping condition not met after max iterations
            if i < 0:
                print('Max iteration reached, no breakpoint found')
                # Since no solution, put "breakpoint" at the end of the data
                beta = 0.
                gamma = 0.
                psi = np.max(x)
                intercept = np.mean(y)
                break

        if i >= 0:
            print('Found breakpoint at x = ' + str(psi))

        # Store the result of the fit
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.psi = psi
        self.intercept = intercept

    def fit(self, x, y):
        """Fit a piecewise linear regression with a single breakpoint to data.

        Parameters
        ----------
        x : numpy.array
            Training data, independent variable
        y : numpy.array
            Training data, dependent variable

        Returns
        -------
        None
        """

        self.train_x = x
        self.train_y = y
        self.__find_psi__()

    def pred(self, x):
        """Predict outputs for test inputs using a previously fit piecewise
           linear regression.

        Parameters
        ----------
        x : numpy.array
            Test data for prediction, independent variable

        Returns
        -------
        numpy.array
            Result of prediction on test data
        """

        y = self.alpha * x + self.beta * self.__Us__(x) + self.gamma * self.__Vs__(x) + self.intercept
        return y
