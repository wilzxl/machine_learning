from sklearn.neighbors import KNeighborsClassifier
import preprocess as pr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def knn_classifier(train_data, test_data, K):
    test_label = np.hstack(pd.DataFrame.from_csv('gender_submission.csv').values)

    start_time = time.time()
    Acc = np.zeros((K, 1))
    Best = -1.0
    N = 0
    for i in range(1, K+1):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(train_data[0:,1:], train_data[0:,0])
        test_prediction = neigh.predict(test_data[0:,0:])
        Acc[i-1] = np.sum(test_label == test_prediction)*1.0/len(test_label)
        if Acc[i-1] > Best:
            Best = Acc[i-1]
            N = i
            test_pred = test_prediction

    plt.plot(np.arange(1, K+1), Acc)
    plt.xlabel('K', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title("Prediction Accuracy of Different K Value", fontsize=10, fontweight = 'bold')
    print "--- %s seconds ---" % (time.time() - start_time)
    print "When K = ", N
    print "Best Prediction Accuracy for KNN:", Best[0]
    plt.show()

def cross_validation():
    train_path = pr.process_data('train.csv')
    train_df = pd.DataFrame.from_csv(train_path)
    train_data = train_df.values
    test_path = pr.process_data('test.csv')
    test_df = pd.DataFrame.from_csv(test_path)
    test_data = test_df.values
    K = 50
    knn_classifier(train_data, test_data, K)


if __name__ == '__main__':
    cross_validation()

