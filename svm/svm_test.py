
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool
from pprint import pprint
import time
import pickle
import numpy as np
import utils
from svm_multi import SvmMulti
from logger import Logger

MAX_THREADS = 3
DUMP_PATH = None #r'dumps/classifiers__26_10_2017__17_06_26.pkl'

# create logger
logger = Logger()

# load dataset
logger.log('load')
x_train, y_train = utils.load_dataset(split=False, allowed_classes=None)

# flat dataset
logger.log('flat')
x_train_flat = utils.flat_dataset(x_train)

# split to train and test datasets
logger.log('split data')
x_train, x_test, y_train, y_test = train_test_split(x_train_flat, y_train, test_size=0.3, random_state=0)

classifiers = list()
if DUMP_PATH is None:
    # create svm multi trainer
    logger.log('create multi svn trainer')
    trainer = SvmMulti(logger, x_train, y_train, x_test, y_test)

    # create thread pool
    thread_pool = ThreadPool(MAX_THREADS)

    # train classifiers
    logger.log('train classifiers')
    thread_pool.map(trainer.get_classifier_for_class, [(i, classifiers) for i in range(1, 26)])
    thread_pool.close()
    thread_pool.join()

    # store classifiers in file
    logger.log('store classifiers if file')

    dump_file_lines = list()
    for classifier in classifiers:
        classifier_line = '\t'.join([str(classifier[0]), str(pickle.dumps(classifier[1])), str(classifier[2])])
        dump_file_lines.append(classifier_line)

    file_name = 'dumps/classifiers__{timestamp}.pkl'.format(timestamp=time.strftime('%d_%m_%Y__%H_%M_%S'))
    with open(file_name, 'wt') as dump_file:
        dump_file.write('\n'.join(dump_file_lines))

else:
    with open(DUMP_PATH, 'rt') as dump_file:
        dump_file_content = dump_file.read()

    dump_file_lines = dump_file_content.split('\n')
    for line in dump_file_lines:
        line_parts = line.split('\t')
        classifiers.append([int(line_parts[0]), pickle.loads(eval(line_parts[1])), float(line_parts[2])])

# test against all classifiers
logger.log('test classifiers joint accuracy')
correct_predictions_counter = 0
# confusion_matrix = np.zeros((25, 25))
for i in range(len(x_test)):
    max_prob = 0
    max_prob_class = 0
    for j in range(len(classifiers)):
        prediction = classifiers[j][1].predict([x_test[i]])

        new_pred = prediction[0] if classifiers[j][2] > 0.5 else 1-prediction[0]
        new_prob = classifiers[j][2] if classifiers[j][2] > 0.5 else 1-classifiers[j][2]

        if new_pred*new_prob > max_prob:
            max_prob = new_pred*new_prob
            max_prob_class = classifiers[j][0]

    if max_prob_class == y_test[i]:
        correct_predictions_counter += 1

# calculate overall accuracy
logger.log('overall accuracy: {acc}'.format(acc=float(correct_predictions_counter)/len(x_test)))


