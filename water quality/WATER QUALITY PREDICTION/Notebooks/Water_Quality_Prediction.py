import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import pickle

from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings(action="ignore")
plt.style.use(["seaborn-bright","dark_background"])

data = pd.read_csv("water_potability.csv")
data.head()

data.describe(include="all")

for i in data.columns:
    per = data[i].isnull().sum()/data.shape[0]
    print("Feature {} has {}% data missing".format(i,round(per*100,2)))

mean1 = data["ph"].mean()
mean2 = data["Sulfate"].mean()
mean3 = data["Trihalomethanes"].mean()

data["ph"] = data["ph"].fillna(mean1)
data["Sulfate"] = data["Sulfate"].fillna(mean2)
data["Trihalomethanes"] = data["Trihalomethanes"].fillna(mean3)

plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True, cmap = "spring")
plt.show()

X = data.drop(columns=["Potability"])
y = data["Potability"]

smote = SMOTE()

X_sample, y_sample = smote.fit_resample(X, y)

print('Original dataset \n',y.value_counts())
print('Resample dataset \n', y_sample.value_counts())

x_train, x_test, y_train, y_test = train_test_split(X_sample,y_sample,test_size=0.15, random_state=101)

model = LogisticRegression()
model.fit(x_train, y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

print(classification_report(y_train,train_pred))
print(classification_report(y_test,test_pred))

print(confusion_matrix(y_train,train_pred))
print(confusion_matrix(y_test,test_pred))

models = []
models.append(("LR", LogisticRegression()))
models.append(("DT", DecisionTreeClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("ET", ExtraTreesClassifier()))
models.append(("GB", GradientBoostingClassifier()))
models.append(("SVC", SVC()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("GNB", GaussianNB()))

for name, model in models:
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print(name)
    print(classification_report(y_train, train_pred))
    print(classification_report(y_test, test_pred))

    print(confusion_matrix(y_train, train_pred))
    print(confusion_matrix(y_test, test_pred))
    print('')

model1 = GradientBoostingClassifier()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier()
model4 = ExtraTreesClassifier()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)


pred_prob1 = model1.predict_proba(x_test)
pred_prob2 = model2.predict_proba(x_test)
pred_prob3 = model3.predict_proba(x_test)
pred_prob4 = model4.predict_proba(x_test)
[19]
from sklearn.metrics import roc_curve

fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)

random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])


print(auc_score1,",", auc_score2,"," ,auc_score3,",",auc_score4)

plt.plot(fpr1, tpr1, linestyle='--',color='r', label='Gradient Boosting')
plt.plot(fpr2, tpr2, linestyle='--',color='yellow', label='Decision Tree')
plt.plot(fpr3, tpr3, linestyle='--',color='c', label='Random Forest')
plt.plot(fpr4, tpr4, linestyle='--',color='lime', label='Extra Tree')
plt.plot(p_fpr, p_tpr, linestyle='-', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show();

model = ExtraTreesClassifier()
model.fit(x_train, y_train)

pickle_out = open("water_potability.pkl","wb")
pickle.dump(model,pickle_out)
loaded_model = pickle.load(open("water_potability.pkl","rb"))
result = loaded_model.score(x_test,y_test)
print(result)