
from sklearn.model_selection import train_test_split
from sklearn import svm
import utils

# load dataset
print('load')
x_train, y_train = utils.load_dataset(False)

# flat dataset
print('flat')
x_train_flat = utils.flat_dataset(x_train)

# split to train and test datasets
print('split data')
x_train, x_test, y_train, y_test = train_test_split(x_train_flat, y_train, test_size=0.3, random_state=0)

# train random forest
print('create classifier')
clf = svm.SVC(kernel='linear', C=1)

print('train classifier')
clf.fit(x_train, y_train)

# test accuracy
print('calculate accuracy')
print(clf.score(x_test, y_test))

print('done')
