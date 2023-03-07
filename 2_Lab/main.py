import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classificator import Classificator
from custom_pca import PCA_method, calc_variance_ratio, get_optimal_n_componets
from matplotlib import pyplot as plt
np.random.seed(777)


   
def preprocess_german_data():
    X = []
    Y = []
    with open("2_Lab/german.data-numeric") as data:
        for row in data:
            string = list(map(int, row.split()))
            X.append(string[:-1])
            Y.append(string[-1] - 1)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y





if __name__ == '__main__':
    # Свои данные
    n1, n2 = 500, 500
    mu1, mu2 = np.array([1,2, 3]),  np.array([2,1,3])
    cov = np.array([[0.5, 1, 0],
                    [1, 2.5, 0],
                    [0, 0, 1]])
    
    x1, y1, x2, y2 = np.random.multivariate_normal(mu1, cov, n1), np.zeros(n1), np.random.multivariate_normal(mu2, cov, n2), np.ones(n2)

    n = 200
    x_train, x_test, y_train, y_test = train_test_split(np.vstack((x1, x2)), np.hstack((y1, y2)), test_size=n, random_state=1234)
    
    classificator = Classificator()
    classificator.fit(x_train, y_train)
    y_res = classificator.predict(x_test)
    wrong_values_indx = np.where(np.not_equal(y_res, y_test))

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1, 3, 1,projection='3d')
    ax1.set_title('Классификатор')
    ax1.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], color=['r' if i else 'b' for i in y_res])

    ax2 = fig.add_subplot(1, 3, 2,projection='3d')
    ax2.set_title('Реальность')
    '''
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕⬛⬛⬛⬛⬛⬛⬛📘📘📕📒📕📘📘📕 
    📕📒📕📘⬛⬛⬛📒📒📒📒📒⬛⬛⬛📕📒📕📘📘📕 
    📕📒📕⬛⬛📒📒📒📒📒📒📒📒📒⬛⬛📒📕📘📘📕 
    📕📒⬛⬛📒📒📒📒📒📒📒📒📒📒📒⬛⬛📕📘📘📕 
    📕📒⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛📕📘📘📕 
    📕⬛📒⬛⬛⬛⬛⬛⬛⬛📒📒⬛⬛⬛⬛⬛⬛📘📘📕 
    📕⬛📒⬛⬛⬛⬛⬛⬛⬛📒📒⬛⬛⬛⬛⬛⬛📘📘📕 
    📕⬛📒⬛⬛⬛⬛⬛⬛⬛📒📒⬛⬛⬛⬛⬛⬛📘📘📕 
    📕⬛📒📒⬛⬛⬛⬛⬛📒📒📒📒⬛⬛⬛⬛⬛📘📘📕 
    📕⬛📒📒📒📒📒📒📒📒📒📒📒📒📒📒📒⬛📘📘📕 
    📕⬛📒📒⬛⬛⬛⬛⬛⬛⬛📒📒📒📒📒📒⬛📘📘📕 
    📕⬛⬛📒⬛⬜⬜⬜⬜⬜⬛⬛📒📒📒📒⬛⬛📘📘📕 
    📕📒⬛📒⬛⬛⬜⬜⬜⬜⬜⬛📒📒📒📒⬛📕📘📘📕 
    📕📒⬛⬛📒⬛⬛⬛⬛⬛⬛📒📒📒📒⬛⬛📕📘📘📕 
    📕📒📕⬛⬛📒📒📒📒📒📒📒📒📒⬛⬛📒📕📘📘📕 
    📕📒📕📘⬛⬛⬛📒📒📒📒📒⬛⬛⬛📕📒📕📘📘📕 
    📕📒📕📘📘📕⬛⬛⬛⬛⬛⬛⬛📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕 
    📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕📒📕📘📘📕
    '''
    ax2.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], color=['g' if np.isin(indx, wrong_values_indx) else ['r', 'b'][res] for indx, res in enumerate(y_res)])
    ax2 = fig.add_subplot(1,3,3)
    
    cm = confusion_matrix(y_test, y_res)
    ConfusionMatrixDisplay(cm, display_labels=["1", "2"]).plot(ax = ax2)
    print(f'm/n P(2|1): {cm[0][1] / (cm[0][0] + cm[0][1])}, P(1|2): {cm[1][0] / (cm[1][0] + cm[1][1])}')
    print(f'С помощью метрики и  P(2|1): {classificator.prob_2_1}, P(1|2): {classificator.prob_1_2}')
    plt.show()


    # Репозиторий
    x, y = preprocess_german_data()
    n = 200
    x_train, x_test, y_train, y_test = train_test_split(PCA_method(x, 10), y, test_size=n, random_state=1234)

    classificator = Classificator()
    classificator.fit(x_train, y_train)
    y_res = classificator.predict(x_test)

    cm = confusion_matrix(y_test, y_res)
    ConfusionMatrixDisplay(cm, display_labels=["1", "2"]).plot()
    print(f'm/n P(2|1): {cm[0][1] / (cm[0][0] + cm[0][1])}, P(1|2): {cm[1][0] / (cm[1][0] + cm[1][1])}')
    print(f'С помощью метрики и  P(2|1): {classificator.prob_2_1}, P(1|2): {classificator.prob_1_2}')
    plt.show()
    
