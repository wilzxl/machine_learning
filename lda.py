from sklearn.neighbors import KNeighborsClassifier
import preprocess as pr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

def lda_classifier(train_data, test_data):
    test_label = np.hstack(pd.DataFrame.from_csv('gender_submission.csv').values)

    start_time = time.time()
    clf1 = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    clf1.fit(train_data[0:,1:],train_data[0:,0])
    test_prediction1 = clf1.predict(test_data[0:,0:])
    print "--- %s seconds ---" % (time.time() - start_time)
    print "Prediction Accuracy for LDA (Eigenvalue Decomposition):", np.sum(test_label == test_prediction1)*1.0/len(test_label)

    # start_time = time.time()
    # clf2 = LinearDiscriminantAnalysis()
    # clf2.fit(train_data[0:,1:],train_data[0:,0])
    # test_prediction2 = clf2.predict(test_data[0:,0:])
    # print "--- %s seconds ---" % (time.time() - start_time)
    # print "Prediction Accuracy for LDA (SVD):", np.sum(test_label == test_prediction2)*1.0/len(test_label)


def cross_validation():
    train_path = pr.process_data('train.csv')
    train_df = pd.DataFrame.from_csv(train_path)
    train_data = train_df.values
    test_path = pr.process_data('test.csv')
    test_df = pd.DataFrame.from_csv(test_path)
    test_data = test_df.values
    lda_classifier(train_data, test_data)


if __name__ == '__main__':
    cross_validation()