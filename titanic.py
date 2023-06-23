import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"D:\VCET\Data Science Internship\Datasets\titanic\train.csv")
test_data = pd.read_csv(r"D:\VCET\Data Science Internship\Datasets\titanic\test.csv")

rf = RandomForestClassifier(random_state=1)
lr = LogisticRegression(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state=0)
gbm = GradientBoostingClassifier(n_estimators=10)

df['Age'].fillna((df['Age'].mean()), inplace=True)

drop_list = ['PassengerId','Name','Ticket','Embarked']
x = df.drop(drop_list,axis=1)
y = df['Embarked']

#Encoding Features
le = LabelEncoder()

x['Sex'] = le.fit_transform(x['Sex'])
x['Cabin'] = le.fit_transform(x['Cabin'])

y =le.fit_transform(y)

#feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_imp = pd.Series(model.feature_importances_,index=x.columns)
feat_imp.nlargest(4).plot(kind='barh')
plt.show()

#Balancing
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=0)
print('Before:')
print(y.value_counts())
x_over, y_over = ros.fit_resample(x,y)
print('After:')
print(y_over.value_counts())
x_over, y_over = ros.fit_resample(x,y)

#Traning and Testing Of dataset
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.3)

#Random Forest
rf.fit(x_train,y_train)

y_pred1 = rf.predict(x_test)
print(accuracy_score(y_test,y_pred1))


#Decision Tree
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)  
y_pred2= classifier.predict(x_test) 
print(accuracy_score(y_test,y_pred2))

#Logistic Regression
lr.fit(x_train, y_train)

y_pred3 = lr.predict(x_test)
print(accuracy_score(y_test,y_pred3))

#Gradient boosting algorithm
gbm = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5 )
gbm.fit(x_train,y_train)
y_pred4 = lr.predict(x_test)
print(accuracy_score(y_test,y_pred4))