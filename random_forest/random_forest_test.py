
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import utils

# load dataset
print('load')
x_train, y_train = utils.load_dataset(False)

# flat dataset
print('flat')
x_train_flat = utils.flat_dataset(x_train)

# split to train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x_train_flat, y_train, test_size=0.3, random_state=0)

# train random forest
print('create random forest')
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
print('train random forest')
# clf = clf.fit(x_train_flat, y_train)
scores = clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

# test accuracy
# clf.

print('done')
