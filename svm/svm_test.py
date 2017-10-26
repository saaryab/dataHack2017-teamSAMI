
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool
from pprint import pprint
import time
import pickle
import utils
from svm_multi import SvmMulti

MAX_THREADS = 3
DUMP_PATH = None


# load dataset
print('load')
x_train, y_train = utils.load_dataset(split=False, allowed_classes=None)

# flat dataset
print('flat')
x_train_flat = utils.flat_dataset(x_train)

# split to train and test datasets
print('split data')
x_train, x_test, y_train, y_test = train_test_split(x_train_flat, y_train, test_size=0.3, random_state=0)

classifiers = list()
if DUMP_PATH is None:
    # create svm multi trainer
    print('create multi svn trainer')
    trainer = SvmMulti(x_train, y_train, x_test, y_test)

    # create thread pool
    thread_pool = ThreadPool(MAX_THREADS)

    # train classifiers
    print('train classifiers')
    thread_pool.map(trainer.get_classifier_for_class, [(i, classifiers) for i in range(1, 26)])
    thread_pool.close()
    thread_pool.join()

    # store classifiers in file
    print('store classifiers if file')

    dump_file_lines = list()
    for classifier in classifiers:
        classifier_line = '\t'.join([str(classifier[0]), str(pickle.dumps(classifier[1])), str(classifier[2])])
        dump_file_lines.append(classifier_line)

    file_name = 'dumps/classifiers__{timestamp}.pkl'.format(timestamp=time.strftime('%d_%m_%Y__%H_%M_%S'))
    with open(file_name, 'wt') as dump_file:
        dump_file.writelines(dump_file_lines)

else:
    with open(DUMP_PATH, 'rt') as dump_file:
        dump_file_content = dump_file.read()

    dump_file_lines = dump_file_content.replace('\r', '').split('\n')
    for line in dump_file_lines:
        line_parts = line.split('\t')
        classifiers.append([int(line_parts[0]), pickle.loads(line_parts[1]), float(line_parts[2])])

# test against all classifiers
print('test classifiers joint accuracy')
correct_predictions_counter = 0
for i in range(len(x_test)):
    max_prob = 0
    max_prob_class = 0
    for j in range(len(classifiers)):
        prediction = classifiers[j][1].predict_proba([x_test[i]])
        pprint(prediction)
        if prediction[0] > max_prob:
            max_prob = prediction[0]
            max_prob_class = classifiers[j][0]

    if max_prob_class == y_test[i]:
        correct_predictions_counter += 1

# calculate overall accuracy
print('overall accuracy: {acc}'.format(acc=float(correct_predictions_counter)/len(x_test)))


# # split to train and test datasets
# print('split data')
# x_train, x_test, y_train, y_test = train_test_split(x_train_flat, y_train, test_size=0.3, random_state=0)
# print('x_train:', len(x_train))
#
# # train random forest
# print('create classifier')
# clf = svm.SVC(kernel='linear', C=1, max_iter=100000)
#
# print('train classifier')
# clf.fit(x_train, y_train)
#
# # test accuracy
# print('calculate accuracy')
# print(clf.score(x_test, y_test))
#
# print('done')
