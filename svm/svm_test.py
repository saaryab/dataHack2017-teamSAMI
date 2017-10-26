
from sklearn.model_selection import train_test_split
from sklearn import svm
import utils
from svm_multi import SvmMulti

# load dataset
print('load')
x_train, y_train = utils.load_dataset(split=False, allowed_classes=None)

# flat dataset
print('flat')
x_train_flat = utils.flat_dataset(x_train)

# create svm multi trainer
trainer = SvmMulti(x_train_flat, y_train)

classifiers = list()
for i in range(1, 26):
    classifiers.append(trainer.get_classifier_for_class(i))

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
