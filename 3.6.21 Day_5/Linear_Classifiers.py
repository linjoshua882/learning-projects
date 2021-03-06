from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)

"""
"""

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))

"""
"""

lr = LogisticRegression()
lr.fit(X, y)

review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])

"""
"""

classifiers = [LogisticRegression(), LinearSVC(),
               SVC(), KNeighborsClassifier()]

for c in classifiers:
    c.fit(X, y)

plot_4_classifiers(X, y, classifiers)
plt.show()

"""
"""

model.coef_ = np.array([[2,2]])
model.intercept_ = np.array([0])

plot_classifier(X,y,model)

num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)

"""
"""

def my_loss(w):
    s = 0
    for i in range(y.size):
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s

w_fit = minimize(my_loss, X[0]).x
print(w_fit)

lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)

"""
"""

def log_loss(raw_model_output):
   return np.log(1+np.exp(-raw_model_output))
def hinge_loss(raw_model_output):
   return np.maximum(0,1-raw_model_output)

grid = np.linspace(-2,2,1000)
plt.plot(grid, log_loss(grid), label='logistic')
plt.plot(grid, hinge_loss(grid), label='hinge')
plt.legend()
plt.show()

"""
"""

def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

w_fit = minimize(my_loss, X[0]).x
print(w_fit)

lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)

"""
"""

train_errs = list()
valid_errs = list()

for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)
    
    train_errs.append( 1.0 - lr.score(X_train, y_train) )
    valid_errs.append( 1.0 - lr.score(X_valid, y_valid) )
    
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()
