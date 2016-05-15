import numpy as np
import scipy.io as sio
from sklearn import preprocessing, cross_validation

from ensemble_regressor import EnsembleRegressor
from plotting_tools import plot_regression_results, plot_y_e_correlation
from regression_datasets import DatasetFactory, dataset_list


def make_ensemble(dataset, mat_filename='ensemble.mat', plotting=True):
    '''
    construct_ensemble splits the dataset into train and 'test'. The ensemble regressors are trained on the training
    set. The test set is saved to the mat file to be used by s
    :param dataset: a dataset object created by DatasetFactory
    :param mat_filename: name of the mat file to save the results to
    :param plotting: plots results
    '''
    if len(dataset.target) < 20000:
        (test_size,train_size) = (0.75, 0.25)
    else:
        (test_size, train_size) = (0.5, 0.5)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        dataset.data,  # preprocessing.scale(dataset.data)
        dataset.target, random_state=0, test_size=test_size, train_size=train_size)

    # Prepare ensemble regressors
    ensemble = EnsembleRegressor(verbose=False)

    n = len(y_train)
    m = ensemble.regressor_count
    if n < m*100:
        overlap = n
        samples_per_regressor = n
    else:  # we have enough samples to be training on different parts of the dataset
        overlap = 0 # n*0.1 # 10% overlap
        samples_per_regressor = (n + m * overlap) // m  # '//' is python operator for floor of n/m

    print "Training set size: %d with %d attributes" % X_train.shape
    print "Each regressor is trained on %d samples" % samples_per_regressor
    print "Test set size: %d" % len(y_test)

    ensemble.fit(X_train, y_train, samples_per_regressor=samples_per_regressor, regressor_overlap=overlap)
    scores = ensemble.score(X_train, y_train)
    MSEs = ensemble.mean_squared_error(X_train, y_train)

    for i, regr in enumerate(ensemble.regressors):
        print '## ' + str(i) + '. ' + regr.__class__.__name__ + ':'
        print regr

        print '\tMSE: %.2f' % MSEs[i]
        print '\tVariance score (R^2): %.2f\n' % scores[i]

    # create predictions matrix on the test set
    Z = ensemble.predict(X_test)

    # Set aside 200 samples from the test set, so that the supervised ensemble learners can use them as a training set
    # X_ensemble_train, X_ensemble_test, y_ensemble_train, y_ensemble_test = cross_validation.train_test_split(
    #     X_test, y_test, random_state=0, train_size=200)
    # Z = ensemble.predict(X_ensemble_test)
    # Z_train = ensemble.predict(X_ensemble_train)

    # Set aside 200 samples as a training set for the supervised ensemble learners
    Z_train, Z, y_ensemble_train, y_ensemble_test = \
        cross_validation.train_test_split(Z.T, y_test, random_state=0, train_size=200)
    Z_train = Z_train.T
    Z = Z.T


    sio.savemat(mat_filename, {
        'names': ensemble.regressor_labels,
        'Z': Z, 'y': y_ensemble_test,
        'Ztrain': Z_train, 'ytrain': y_ensemble_train,
        'samples_per_regressor': samples_per_regressor,
        'regressor_samples_overlap': overlap,
        'Ey': np.mean(y_ensemble_test),  # np.mean(dataset.target),
        'Ey2': np.mean(y_ensemble_test ** 2)  # np.mean(dataset.target ** 2)
    })

    if plotting:
        plot_regression_results(ensemble, Z, y_ensemble_test)
        plot_y_e_correlation(ensemble, Z, y_ensemble_test)


def main():
    # Prepare dataset
    # make_ensemble(DatasetFactory.nasdaq_index(), "auto/auto_NASDAQ_index.mat")
    # make_ensemble(DatasetFactory.flights(origin_airport='longhaul'), "auto/auto_flights_longhaul.mat", plotting=False)
    # make_ensemble(DatasetFactory.blockbuster(), "auto/auto_blockbuster.mat")
    # make_ensemble(DatasetFactory.boston())

    # for name,func in dataset_list.iteritems():
    #     print(name)
    #     dataset = func()
    #     make_ensemble(dataset, "auto_mlp5_change_h_num/auto_" + name + ".mat", plotting=False)
    make_ensemble(DatasetFactory.friedman1(), "auto/auto_friedman1_new.mat")
    make_ensemble(DatasetFactory.friedman2(), "auto/auto_friedman2_new.mat")
    make_ensemble(DatasetFactory.friedman3(), "auto/auto_friedman3_new.mat")

    print('Done.')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    # except IOError, e:  # catch closing of pipes
    #     if e.errno != errno.EPIPE:
    #         raise e
