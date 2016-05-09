import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn import datasets, cross_validation
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

# Prepare dataset

# dataset = datasets.load_boston()
dataset = datasets.load_diabetes()
# dataset = datasets.fetch_mldata('regression-datasets abalone', data_home='~/Downloads/Datasets/sklearn/')
# dataset = datasets.fetch_mldata('regression-datasets servo', data_home='~/Downloads/Datasets/sklearn/')

# class DataSet:
#     data = None
#     target = None
#
# dataset = DataSet()
# (dataset.data, dataset.target) = datasets.make_friedman1(n_samples=1000)
# (dataset.data, dataset.target) = datasets.make_friedman2(n_samples=1000)
# (dataset.data, dataset.target) = datasets.make_friedman3(n_samples=1000)

# boston_X = boston_scaled #[:, np.newaxis, 2] # the np.newaxis makes it into a column vector
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        preprocessing.scale(dataset.data),
                                        dataset.target, test_size=0.75, random_state=0)
# boston_X = boston.data
# boston_X_scaled = preprocessing.scale(boston_X)
# boston_X_train = boston_X[:-20]
# boston_X_train_scaled = boston_X_scaled[:-20]
# boston_X_test = boston_X[-20:]
# boston_X_test_scaled = boston_X_scaled[-20:]
# boston_y_train = boston.target[:-20]
# boston_y_test = boston.target[-20:]

# Prepare ensemble regressors

regressors = (
    linear_model.LinearRegressi        on(fit_intercept=True),
    Pipeline(
        [('poly', PolynomialFeatures(degree=2)),
         ('linear', linear_model.LinearRegression(fit_intercept=False))]
    ),
    linear_model.Ridge(alpha=.1, fit_intercept=True),
    linear_model.RidgeCV(alphas=[.01, .1, .3, .5, 1], fit_intercept=True),
    linear_model.Lasso(alpha=1, fit_intercept=True),
    linear_model.LassoCV(n_alphas=100, fit_intercept=True),
    linear_model.ElasticNet(alpha=1),
    linear_model.ElasticNetCV(n_alphas=100, l1_ratio=.5),
    linear_model.OrthogonalMatchingPursuit(),
    linear_model.BayesianRidge(),
    linear_model.ARDRegression(),
    linear_model.SGDRegressor(),
    linear_model.PassiveAggressiveRegressor(loss='squared_epsilon_insensitive'),
    linear_model.RANSACRegressor(),
    LinearSVR(max_iter=1e4, fit_intercept=True, loss='squared_epsilon_insensitive', C=0.5),
    SVR(max_iter=1e4, kernel='poly', C=1, degree=4),
    SVR(max_iter=1e4, kernel='rbf', C=1, gamma=0.1),
    SVR(kernel='linear', C=1),
    SVR(kernel='linear', C=0.5),
    SVR(kernel='linear', C=0.1),
    DecisionTreeRegressor(max_depth=5),
    DecisionTreeRegressor(max_depth=4),
    DecisionTreeRegressor(max_depth=None),
    RandomForestRegressor(n_estimators=100),
    AdaBoostRegressor(learning_rate=0.9, loss='square'),
    BaggingRegressor(),
)

# regr = linear_model.LinearRegression()
start = time.time()
for i, regr in enumerate(regressors):
    print '## ' + str(i) + '. ' + regr.__class__.__name__ + ':'
    print regr

    # if type(regr) in [linear_model.LinearRegression, linear_model.Ridge, LinearSVR]:
    #     X_train = boston_X_train
    #     X_test = boston_X_test
    # else:
    #     X_train = boston_X_train_scaled
    #     X_test = boston_X_test_scaled

    regr.fit(X_train, y_train)

    if type(regr) in [linear_model.LinearRegression, linear_model.Ridge, LinearSVR]:
        print '\tCoefficients: ', ', '.join(['%.2f' % f for f in regr.coef_])

    if hasattr(regr, 'alphas_'):
        print '\tAlphas: ', ', '.join(['%.2f' % f for f in regr.alphas_])

    print '\tMSE: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2)
    print '\tVariance score (R^2): %.2f\n' % regr.score(X_test, y_test)

print 'Total running time: %.2f' % (time.time() - start)

# Plotting and saving to mat file
from cycler import cycler

color_cycler = cycler('c', 'rgbmy')
marker_style_cycler = cycler('marker', 'xsovd.')
style_cycler = marker_style_cycler * color_cycler * 2
sty = iter(style_cycler)

plt.figure(figsize=(15, 10))

# Output values
regressor_labels = []
Z = np.ndarray((len(regressors), len(y_test)))
Z_train = np.ndarray((len(regressors), len(y_train)))

for i, regr in enumerate(regressors):
    # s = sty.next()
    # zip the real and predicted values together, sort them, and unzip them
    regressor_labels.append(str(regr))
    Z[i, :] = regr.predict(X_test)
    Z_train[i, :] = regr.predict(X_train)

    points = zip(*sorted(zip(y_test, Z[i, :])))
    plt_x, plt_y = list(points[0]), list(points[1])
    # plt.scatter(boston_y_test, regr.predict(boston_X_test_scaled), label=str(regr), **sty.next())
    plt.plot(plt_x, plt_y, label=regressor_labels[i], **sty.next())

# print (boston_y_test, regr.predict(boston_X_test_scaled), str(regr), a)
# plt.scatter(boston_y_test, regressors[4].predict(boston_X_test_scaled), label=str(regressors[4]), color='r')
# plt.scatter(boston_y_test, regressors[5].predict(boston_X_test_scaled), label=str(regressors[5]), color='g')
# plt.scatter(boston_y_test, regressors[6].predict(boston_X_test_scaled), label=str(regressors[6]), color='b',marker='x')
plt.plot(y_test, y_test, color='black', linewidth=2, label='truth', marker='d')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.15, 1))
# plt.xticks(())
# plt.yticks(())
plt.tight_layout()

sio.savemat('ensemble.mat', {
    'names': regressor_labels,
    'Z': Z, 'y': y_test,
    'Ztrain': Z_train, 'ytrain': y_train
    })

plt.show()
