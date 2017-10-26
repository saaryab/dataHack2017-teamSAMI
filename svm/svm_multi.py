
from sklearn import svm
import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_val_score
import threading


class SvmMulti:

    def __init__(self, logger, full_x_train, full_y_train, x_test, y_test):
        self.__x_train = full_x_train
        self.__y_train = full_y_train
        self.__x_test = x_test
        self.__y_test = y_test
        self.__logger = logger
        self.__classifiers_list_lock = threading.Lock()

    @staticmethod
    def __transform_dataset_for_class(dataset, class_name):

        class_y_train = [
            1 if dataset[i] == class_name else 0 for i in range(len(dataset))
        ]

        return class_y_train

    def get_classifier_for_class(self, params):
        # extract params
        class_name = params[0]
        classifiers_list = params[1]

        # transform the y train tags for the given class
        self.__logger.log('transform y train for {class_name}'.format(class_name=class_name))
        class_y_train = self.__transform_dataset_for_class(self.__y_train, class_name)

        # create classifier for the given class
        self.__logger.log('build classifier for {class_name}'.format(class_name=class_name))
        clf = svm.SVC(kernel='linear', C=1, max_iter=100000)

        # train classifier and test performance
        self.__logger.log('train classifier of {class_name}'.format(class_name=class_name))
        clf.fit(self.__x_train, class_y_train)

        # transform the y test tags for the given class
        self.__logger.log('transform y train for {class_name}'.format(class_name=class_name))
        class_y_test = self.__transform_dataset_for_class(self.__y_test, class_name)

        # calculate classifier score
        self.__logger.log('calculate score of {class_name}'.format(class_name=class_name))
        classifier_score = clf.score(self.__x_test, class_y_test)
        self.__logger.log('score of {class_name} is: {classifier_score}'.format(
            class_name=class_name, classifier_score=classifier_score
        ))

        # print confusion matrix
        # y_pred = clf.predict(self.__x_test)
        # self.__logger.log('confusion matrix of {class_name}: {confusion_matrix}'.format(
        #     class_name=class_name, confusion_matrix=confusion_matrix(class_y_test, y_pred))
        # )

        # calculate confusion matrix
        confusion_matrix_fp = np.zeros(25)
        predictions = clf.predict(self.__x_test)
        correct_predictions = 0
        for i in range(len(self.__x_test)):
            if predictions[i] != class_y_test[i]:
                if predictions[i] == 1:
                    confusion_matrix_fp[self.__y_test[i] - 1] += 1
            else:
                correct_predictions += 1

        self.__logger.log('conf. matrix accuracy score of {class_name}: {score}'.format(
            class_name=class_name, score=float(correct_predictions)/len(self.__x_test))
        )
        self.__logger.log('[{class_name}] FP (say it was him and wrong): {conf_fp}'.format(
            class_name=class_name, conf_fp=confusion_matrix_fp)
        )

        # add classifier info
        with self.__classifiers_list_lock:
            classifiers_list.append([class_name, clf, classifier_score])
