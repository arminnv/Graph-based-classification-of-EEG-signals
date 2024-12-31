import numpy as np
from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from feature_extraction import X, y
print(X.shape, y.shape)
"""
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_val)
print(accuracy_score(y_val, gnb_predictions))

rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)

"""

degree = [1, 2, 3, 4, 5, 6, 7]
C = [0.01, 0.1,0.5, 0.8, 0.9, 1,1.1, 1.2, 1.5, 4, 8, 10, 100]
gamma = [0.01, 1, 10, 50]

#lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
#lm.fit(X_train, y_train)

#accuracy = accuracy_score(y_val, lm.predict(X_val))
#print("log reg: ", accuracy)
best = 0.0
c_best = 0
d_best = 0
max = 0
for c in C:
    for d in degree:
        poly = svm.SVC(kernel='poly', degree=d, C=c)
        scores = cross_val_score(poly, X, y, cv=11)
        print(scores)
        poly_accuracy = np.mean(scores)
        """
        poly_pred = poly.predict(X_train)
        poly_accuracy = accuracy_score(y_train, poly_pred)
        print(f"acc train= {poly_accuracy}, c = {c}, d = {d}")
        poly_pred = poly.predict(X_val)
        poly_accuracy = accuracy_score(y_val, poly_pred)
        print(f"acc = {poly_accuracy}, c = {c}, d = {d}")
        """
        if poly_accuracy > best:
            best = poly_accuracy
            c_best = c
            d_best = d
            max = np.max(scores)
print(f"c = {c_best}, d = {d_best}")
print(f"highest accuracy: {best}")
print(max)

"""
best = 0.0
c_best = 0
g_best = 0

for c in C:
    for g in gamma:
        rbf = svm.SVC(kernel='rbf', gamma=g, C=c).fit(X_train, y_train)
        rbf_pred = rbf.predict(X_val)
        rbf_accuracy = accuracy_score(y_val, rbf_pred)
        print(f"acc = {rbf_accuracy}, c = {c}, g = {g}")
        if rbf_accuracy > best:
            best = rbf_accuracy
            c_best = c
            g_best = g


print(f"c = {c_best}, g = {g_best}")
"""

"""
poly_pred = poly.predict(X_train)
rbf_pred = rbf.predict(X_train)
rbf_accuracy = accuracy_score(y_train, rbf_pred)
rbf_f1 = f1_score(y_train, rbf_pred, average='weighted')
poly_accuracy = accuracy_score(y_train, poly_pred)
poly_f1 = f1_score(y_train, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
poly_pred = poly.predict(X_train)
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
"""
