#!/usr/bin/env python
# coding: utf-8

# In[277]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn import metrics
from collections import Counter
import warnings


# In[2]:


warnings.simplefilter(action="ignore")


# In[3]:


# pd.set_option("display.max_rows", None)


# In[4]:


df = pd.read_csv("D:\programming\Machine Learning/Bank_Personal_Loan.csv")


# In[5]:


df


# In[6]:


# pd.set_option("display.max_rows")


# In[7]:


df.keys()


# In[8]:


df.columns.to_list()


# In[9]:


df.info()


# <div style=" font-size:22px; line_height:160%">
# The type of one of the columns is object.<br/>
# In addition, column names should have underscore_score instead of space for better use and addressing<br/>
# 

# In[10]:


# تبدیل رشته‌های ستون با تایپ آبجکت به اعداد اعشاری 

list_CCAvg = []

for item in df.CCAvg.values:
    item = list(item)
    i = 0
    while i < len(item):
        if item[i] == "/":
            item[i] = "."
            item = float("".join(item))
            list_CCAvg.append(item)
            break
        i += 1


df["CCAvg"] = list_CCAvg
# df["CCAvg"] = pd.dfFrame({"CCAvg": list_CCAvg})


# In[11]:


# تبدیل فاصله (اسپیس) بین نام ستون‌ها به آندراسکور ــ

df.columns = df.columns.str.replace(' ', '_')


# In[12]:


df


# In[13]:


df.info()


# <div style=" font-size:22px; line_height:160%">
# The above information shows that we do not have missing values.

# In[14]:


df.describe()


# <div style=" font-size:22px; line_height:160%">
# In the experience feature, we have a negative number, which is very irrational. We will fix it in the future.

# In[15]:


df.nunique()


# <div style=" font-size:22px; line_height:160%">
# In analysis and prediction, it is practically meaningless to use the ID column of persons.<br/>
# So we delete it from the dataset.

# In[16]:


df = df.drop("ID", axis=1)


# In[17]:


# Define a function to draw a dot plot

def one_scatter(df_name, x_ax_name, y_ax_name):
    scatter_name = f"{y_ax_name}-{x_ax_name}"
    fig_output_name = scatter_name
    plt.figure(figsize=(10,4), dpi=80)
    plt.title(f"{x_ax_name} - {y_ax_name}\n", fontsize=30 )
    scatter_name = plt.scatter(df_name[x_ax_name], df_name[y_ax_name])
    scatter_name.axes.tick_params(gridOn=True, size=12, labelsize=10)
    plt.xlabel(f"\n{x_ax_name}", fontsize=20)
    plt.ylabel(f"{y_ax_name}\n", fontsize=20)
    plt.xticks(rotation=90)


# In[18]:


# Definition of the function to plot the count of the number of categories of each feature.

def count_plot(df_name, column_name):

    plt.figure(figsize=(20, 5), dpi=90)
    ax = sns.countplot(x=column_name, data=df)
    ax.bar_label(ax.containers[0], fontsize=10)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(f"\n{column_name}", fontsize=20)
    plt.ylabel("count",fontsize=20)
    plt.title(f"Count of {column_name}", fontsize=30)
    plt.grid()


# In[19]:


one_scatter(df, df.columns[0], df.columns[8])


# In[20]:


count_plot(df, df.columns[0])


# In[21]:


one_scatter(df, df.columns[1], df.columns[8])


# In[22]:


count_plot(df, df.columns[1])


# In[23]:


one_scatter(df, df.columns[2], df.columns[8])


# In[24]:


count_plot(df, df.columns[2])


# In[25]:


one_scatter(df, df.columns[3], df.columns[8])


# In[26]:


df.sort_values(by="ZIP_Code")


# In[27]:


# We delete that one outlier data


# In[28]:


df = df[df["ZIP_Code"]>10000]


# In[29]:


count_plot(df, df.columns[3])


# In[30]:


one_scatter(df, df.columns[4], df.columns[8])


# In[31]:


count_plot(df, df.columns[4])


# In[32]:


one_scatter(df, df.columns[5], df.columns[8])


# In[33]:


count_plot(df, df.columns[5])


# In[34]:


one_scatter(df, df.columns[6], df.columns[8])


# In[35]:


count_plot(df, df.columns[6])


# In[36]:


one_scatter(df, df.columns[7], df.columns[8])


# In[37]:


count_plot(df, df.columns[7])


# In[38]:


one_scatter(df, df.columns[9], df.columns[8])


# In[39]:


count_plot(df, df.columns[9])


# In[40]:


one_scatter(df, df.columns[10], df.columns[8])


# In[41]:


count_plot(df, df.columns[10])


# In[42]:


one_scatter(df, df.columns[11], df.columns[8])


# In[43]:


count_plot(df, df.columns[11])


# In[44]:


one_scatter(df, df.columns[12], df.columns[8])


# In[45]:


count_plot(df, df.columns[12])


# <div style=" font-size:22px; line_height:160%">
# Obtaining the percentage of negative values in the experience feature compared to the total data:

# In[46]:


df_negatives = df[df.Experience<0]
df_negatives


# In[47]:


df_negatives_percent = (len(df_negatives)/len(df.Experience))*100
print(f"The percentage of negatives Experience is: %{round(df_negatives_percent, 3)} in {len(df.Experience)} samples.")


# In[48]:


sns.pairplot(df)


# In[49]:


df.corr()


# In[50]:


df_corr = df.corr()
df_corr<0.5


# <div style=" font-size:22px; line_height:160%">
# We see two correlations with a high value (higher than 0.5), which we plot their graphs to see what is going on.

# In[51]:


one_scatter(df, df.columns[0], df.columns[1])


# In[52]:


one_scatter(df, df.columns[2], df.columns[5])


# <div style=" font-size:22px; line_height:160%">
# The first correlation is very high (above 90%) and we will remove it later because we want to write the program once before and once after removing it.

# <div style=" font-size:22px; line_height:160%">
# We don't have a negative work history in reality!<br/>
# So we have to think about them...<br/>
# These data constitute one percent of the total data of the dataset.<br/>

# <div style=" font-size:22px; line_height:160%">
# The ways that can be investigated are:<br/>
# 1 zero placing them<br/>
# 2 removing them<br/>
# 3 replacing their values with their absolute values<br/>
# 4 replacing them with the average value of the total data<br/>
# 5. Due to the high correlation with the age feature, remove the work history feature altogether.<br/>

# <div style=" font-size:22px; line_height:160%">
# According to the data pattern and the way they are collected in the dataset, ways 3 and 4 do not seem logical.

# <div style=" font-size:22px; line_height:160%">
# Therefore, we write and compare the program for the other three modes:

# In[53]:


# CASE 1: negative Experience values = 0

# CASE 2: negative Experience values -> eliminate (delete)

# CASE 3: the "Experience" column -> eliminate the column due to high correlation with "Age" column


# <div style="font-size:30px; color:#33FF8A; line-height: 130%">
# CASE 1:<br/>
#     negative Experience values = 0

# In[54]:


df1 = df.copy()


# In[55]:


df1.loc[df1['Experience']<0, 'Experience'] = 0


# In[56]:


df1.sort_values(by='Experience')


# In[57]:


one_scatter(df1, df1.columns[0], df1.columns[1])


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# Logistic Regression (Case 1)

# In[58]:


X = df1.drop("Personal_Loan", axis=1)
y = df1["Personal_Loan"]


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[60]:


logistic_reg = LogisticRegression(solver="liblinear")


# In[61]:


logistic_reg.fit(X_train, y_train)
y_pred_lg = logistic_reg.predict(X_test)


# In[62]:


y_pred_lg


# In[63]:


f1_score_lg_1 = metrics.f1_score(y_test, y_pred_lg)
print("Score (lg_1):", f1_score_lg_1)


# In[64]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_lg)
plt.plot(fpr, tpr, label="data 1")
plt.legend(loc=4)
plt.show()


# In[65]:


y_pred_proba_lg = logistic_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba_lg)
plt.plot(fpr, tpr, label="data 1")
plt.legend(loc=4)
plt.show()


# In[66]:


Counter(logistic_reg.predict(X))


# In[67]:


confusion_matrix(y, logistic_reg.predict(X))


# In[68]:


cm = confusion_matrix(y, logistic_reg.predict(X))

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=logistic_reg.classes_)
disp.plot()
plt.show()


# In[69]:


target_names = ['class 0', 'class 1']
print(classification_report(y, logistic_reg.predict(X), target_names=target_names))


# In[70]:


df_sample = pd.DataFrame({
    "Age": [42],
    "Experience": [16],
    "Income": [30],
    "ZIP_Code": [92037],
    "Family": [3],
    "CCAvg": [1.2],
    "Education": [3.0],
    "Mortgage": [0],
    "Securities_Account": [1],
    "CD_Account": [0],
    "Online": [1],
    "CreditCard": [1],
    "Personal_Loan": [1],
#     "ID": [5071]  # we don't need this feature for prediction
})


# In[71]:


Df = pd.concat([df1, df_sample])
df_sample = Df.iloc[Df.shape[0]-1:]

df_sample


# In[72]:


X_train = df1.drop("Personal_Loan", axis=1)
y_train = df1["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[73]:


logistic_reg.fit(X_train, y_train)
y_pred_1 = logistic_reg.predict(X_test)


# In[74]:


print(y_pred_1)


# In[75]:


# Cross Validation


# In[76]:


for i in range(5, 11):
    kfold_validation = KFold(i, shuffle=True, random_state=0)
    results = cross_val_score(logistic_reg, X, y, cv=kfold_validation)
    results_mean = np.mean(results)
    print(f"For KFold = {i}:")
    print(f"results:\n{results}")
    print(f"mean results:\n{results_mean}")


# <div style=" font-size:22px; line_height:160%">
# The result of cross validation shows that in the dataset, we do not have data that have a significant negative effect on score.

# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# ComplementNB (Case 1)

# In[77]:


Counter(y)


# <div style=" font-size:22px; line_height:160%">
# This indicates that the target data is imbalanced. Because the number of data 0 is much more than the number of data 1.<br/>
# As a result, we use Naive Bayes Compliment.<br/>
# Also, for this reason, the accuracy metric is not a good measure and we should consider other metrics or the confusion matrix.

# In[78]:


X = df1.drop("Personal_Loan", axis=1)
y = df1["Personal_Loan"]#.values.reshape(-1, 1)


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[80]:


clf_NB = ComplementNB()


# In[81]:


clf_NB.fit(X_train, y_train)


# In[82]:


y_pred_NB = clf_NB.predict(X_test)


# In[83]:


print(y_pred_NB)


# In[84]:


f1_score_NB_1 = metrics.f1_score(y_test, y_pred_NB)
print("Score (NB_1):", f1_score_NB_1)


# In[85]:


cm = confusion_matrix(y, clf_NB.predict(X))

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf_NB.classes_)
disp.plot()
plt.show()


# In[86]:


target_names = ['class 0', 'class 1']
print(classification_report(y, clf_NB.predict(X), target_names=target_names))


# In[87]:


X_train = df1.drop("Personal_Loan", axis=1)
y_train = df1["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[88]:


clf_NB.fit(X_train, y_train)
y_pred_2 = clf_NB.predict(X_test)


# In[89]:


print(y_pred_2)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# KNN Algorithm (Case 1)

# In[90]:


X = df1.drop("Personal_Loan", axis=1).values
y = df1["Personal_Loan"]#.values.reshape(-1, 1)


# <div style=" font-size:22px; line_height:160%">
# For X, we wrote a .values that has array mode instead of dataframe mode<br/>
# Because in this case, the calculations are done at a higher speed, the amount of memory used is less, and the amount of CPU usage is less

# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[92]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
# X_train_scaled = pd.DataFrame(X_train_scaling, columns=df.columns.drop("Personal_Loan"))
X_test_scaling = Scaler.transform(X_test)
# X_test_scaled = pd.DataFrame(X_test_scaling, columns=df.columns.drop("Personal_Loan"))


# In[93]:


K = 20
Acc = np.zeros(K)
# print("Acc", Acc)

for i in range(1, K+1):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_scaling, y_train.ravel())
    y_pred = clf.predict(X_test_scaling)
    Acc[i-1] = metrics.f1_score(y_test, y_pred)
    
Acc


# In[94]:


print(np.max(Acc))
print(np.min(Acc))


# <div style=" font-size:22px; line_height:160%">
# Finding the appropriate k for recall score

# In[95]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.recall_score(y_train, y_train_new))
    test_acc.append(metrics.recall_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[96]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="recall",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[97]:


grid_kn.best_params_


# In[98]:


K = 1
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[99]:


f1_score_KNN_1_recall = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_1_recall):", f1_score_KNN_1_recall)


# In[100]:


X_train = df1.drop("Personal_Loan", axis=1).values
y_train = df1["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[101]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
# X_train_scaled = pd.DataFrame(X_train_scaling, columns=df.columns.drop("Personal_Loan"))
X_test_scaling = Scaler.transform(X_test)
# X_test_scaled = pd.DataFrame(X_test_scaling, columns=df.columns.drop("Personal_Loan"))


# In[102]:


clf.fit(X_train_scaling, y_train)
y_pred_3_1 = clf.predict(X_test_scaling)


# In[103]:


print(y_pred_3_1)


# <div style=" font-size:22px; line_height:160%">
# Finding the right k for precision

# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[105]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[106]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.precision_score(y_train, y_train_new))
    test_acc.append(metrics.precision_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[107]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="precision",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[108]:


grid_kn.best_params_


# In[109]:


K = 6
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN_1 = clf.predict(X_test_scaling)


# In[110]:


f1_score_KNN_1_precision = metrics.f1_score(y_test, y_pred_KNN_1)
print("Score (knn_1_precision):", f1_score_KNN_1_precision)


# In[111]:


X_train = df1.drop("Personal_Loan", axis=1).values
y_train = df1["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[112]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[113]:


clf.fit(X_train_scaling, y_train)
y_pred_3_2 = clf.predict(X_test_scaling)


# In[114]:


print(y_pred_3_2)


# <div style=" font-size:22px; line_height:160%">
# Find the appropriate k for the score of f1

# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[116]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[117]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.f1_score(y_train, y_train_new))
    test_acc.append(metrics.f1_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[118]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="f1",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[119]:


grid_kn.best_params_


# In[120]:


K = 5
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN_1 = clf.predict(X_test_scaling)


# In[121]:


f1_score_KNN_1_f1 = metrics.f1_score(y_test, y_pred_KNN_1)
print("Score (knn_1_f1):", f1_score_KNN_1_f1)


# In[122]:


X_train = df1.drop("Personal_Loan", axis=1).values
y_train = df1["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[123]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[124]:


clf.fit(X_train_scaling, y_train)
y_pred_3_3 = clf.predict(X_test_scaling)


# In[125]:


print(y_pred_3_3)


# <div style="font-size:30px; color:#33FF8A; line-height: 130%">
# CASE 2:<br/>
# negative Experience values -> eliminate (delete)

# In[126]:


df2 = df[df['Experience']>=0]
df2.sort_values(by='Experience')


# In[127]:


one_scatter(df2, df2.columns[0], df2.columns[1])


# In[128]:


X = df2.drop("Personal_Loan", axis=1)
y = df2["Personal_Loan"]#.values.reshape(-1, 1)


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# Logistic Regression (Case 2)

# In[130]:


logistic_reg = LogisticRegression(solver="liblinear")


# In[131]:


logistic_reg.fit(X_train, y_train)
y_pred_lg = logistic_reg.predict(X_test)


# In[132]:


y_pred_lg


# In[133]:


f1_score_lg_2 = metrics.f1_score(y_test, y_pred_lg)
print("Score (lg_2):", f1_score_lg_2)


# In[134]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_lg)
plt.plot(fpr, tpr, label="data 1")
plt.legend(loc=4)
plt.show()


# In[135]:


y_pred_proba_lg = logistic_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba_lg)
plt.plot(fpr, tpr, label="data 1")
plt.legend(loc=4)
plt.show()


# In[136]:


Counter(logistic_reg.predict(X))


# In[137]:


confusion_matrix(y, logistic_reg.predict(X))


# In[138]:


cm = confusion_matrix(y, logistic_reg.predict(X))

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=logistic_reg.classes_)
disp.plot()
plt.show()


# In[139]:


target_names = ['class 0', 'class 1']
print(classification_report(y, logistic_reg.predict(X), target_names=target_names))


# In[140]:


# Cross Validation


# In[141]:


for i in range(5, 11):
    kfold_validation = KFold(i, shuffle=True, random_state=0)
    results = cross_val_score(logistic_reg, X, y, cv=kfold_validation)
    print(f"For KFold = {i}:")
    print(f"{results}\n")


# <div style=" font-size:22px; line_height:160%">
# The result of cross validation shows that in the dataset, we do not have data that have a significant negative effect on score.

# In[142]:


df_sample = pd.DataFrame({
    "Age": [42],
    "Experience": [16],
    "Income": [30],
    "ZIP_Code": [92037],
    "Family": [3],
    "CCAvg": [1.2],
    "Education": [3.0],
    "Mortgage": [0],
    "Securities_Account": [1],
    "CD_Account": [0],
    "Online": [1],
    "CreditCard": [1],
    "Personal_Loan": [1],
#     "ID": [5071]
})


# In[143]:


Df = pd.concat([df2, df_sample])
df_sample = Df.iloc[Df.shape[0]-1:]

df_sample


# In[144]:


X_train = df2.drop("Personal_Loan", axis=1)
y_train = df2["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[145]:


logistic_reg.fit(X_train, y_train)
y_pred_1 = logistic_reg.predict(X_test)


# In[146]:


print(y_pred_1)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# ComplementNB (Case 2)

# In[147]:


Counter(y)


# <div style=" font-size:22px; line_height:160%">
# This indicates that the target data is imbalanced.<br/>
# As a result, we use Naive Bayes Compliment.<br/>
# Also, for this reason, the accuracy metric is not a good measure and we should consider other metrics or the confusion matrix.

# In[148]:


X = df2.drop("Personal_Loan", axis=1)
y = df2["Personal_Loan"]#.values.reshape(-1, 1)


# In[149]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[150]:


clf_NB = ComplementNB()


# In[151]:


clf_NB.fit(X_train, y_train)


# In[152]:


y_pred_NB = clf_NB.predict(X_test)


# In[153]:


print(y_pred_NB)


# In[154]:


f1_score_NB_2 = metrics.f1_score(y_test, y_pred_NB)
print("Score (NB_2):", f1_score_NB_2)


# In[155]:


cm = confusion_matrix(y, clf_NB.predict(X))

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf_NB.classes_)
disp.plot()
plt.show()


# In[156]:


target_names = ['class 0', 'class 1']
print(classification_report(y, clf_NB.predict(X), target_names=target_names))


# In[157]:


X_train = df2.drop("Personal_Loan", axis=1)
y_train = df2["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[158]:


clf_NB.fit(X_train, y_train)
y_pred_2 = clf_NB.predict(X_test)


# In[159]:


print(y_pred_2)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# KNN Algorithm (Case 2)

# In[160]:


X = df2.drop("Personal_Loan", axis=1).values
y = df2["Personal_Loan"]#.values.reshape(-1, 1)


# In[161]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[162]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[163]:


K = 20
Acc = np.zeros(K)
print("Acc", Acc)

for i in range(1, K+1):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_scaling, y_train.ravel())
    y_pred = clf.predict(X_test_scaling)
    Acc[i-1] = metrics.f1_score(y_test, y_pred)
    
Acc


# In[164]:


print(np.max(Acc))
print(np.min(Acc))


# <div style=" font-size:22px; line_height:160%">
# Finding the appropriate k for recall score

# In[165]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.recall_score(y_train, y_train_new))
    test_acc.append(metrics.recall_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[166]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="recall",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[167]:


grid_kn.best_params_


# In[168]:


K = 1
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[169]:


f1_score_KNN_2_recall = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_2_recall):", f1_score_KNN_2_recall)


# In[170]:


X_train = df2.drop("Personal_Loan", axis=1).values
y_train = df2["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[171]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[172]:


clf.fit(X_train_scaling, y_train)
y_pred_3_1 = clf.predict(X_test_scaling)


# In[173]:


print(y_pred_3_1)


# <div style=" font-size:22px; line_height:160%">
# Finding the right k for precision

# In[174]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[175]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[176]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.precision_score(y_train, y_train_new))
    test_acc.append(metrics.precision_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[177]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="precision",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[178]:


grid_kn.best_params_


# In[179]:


K = 4
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[180]:


f1_score_KNN_2_precision = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_2_precision):", f1_score_KNN_2_precision)


# In[181]:


X_train = df2.drop("Personal_Loan", axis=1).values
y_train = df2["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[182]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[183]:


clf.fit(X_train_scaling, y_train)
y_pred_3_2 = clf.predict(X_test_scaling)


# In[184]:


print(y_pred_3_2)


# <div style=" font-size:22px; line_height:160%">
# Find the appropriate k for the score of f1

# In[185]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[186]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[187]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.f1_score(y_train, y_train_new))
    test_acc.append(metrics.f1_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[188]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="f1",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[189]:


grid_kn.best_params_


# In[190]:


K = 1
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[191]:


f1_score_KNN_2_f1 = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_2_f1):", f1_score_KNN_2_f1)


# In[192]:


X_train = df2.drop("Personal_Loan", axis=1).values
y_train = df2["Personal_Loan"]
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[193]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[194]:


clf.fit(X_train_scaling, y_train)
y_pred_3_3 = clf.predict(X_test_scaling)


# In[195]:


print(y_pred_3_3)


# <div style="font-size:30px; color:#33FF8A; line-height: 130%">
# CASE 3:<br/>
# the "Experience" column -> eliminate the column due to high correlation with "Age" column

# In[196]:


df3 = df.drop("Experience", axis=1)
df3


# In[197]:


X = df3.drop("Personal_Loan", axis=1)
y = df3["Personal_Loan"]#.values.reshape(-1, 1)


# In[198]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# Logistic Regression (Case 3)

# In[199]:


logistic_reg = LogisticRegression(solver="liblinear")


# In[200]:


logistic_reg.fit(X_train, y_train)
y_pred_lg = logistic_reg.predict(X_test)


# In[201]:


y_pred_lg


# In[202]:


f1_score_lg_3 = metrics.f1_score(y_test, y_pred_lg)
print("Score (lg_3):", f1_score_lg_3)


# In[203]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_lg)
plt.plot(fpr, tpr, label="data 1")
plt.legend(loc=4)
plt.show()


# In[204]:


y_pred_proba_lg = logistic_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba_lg)
plt.plot(fpr, tpr, label="data 1")
plt.legend(loc=4)
plt.show()


# In[205]:


Counter(logistic_reg.predict(X))


# In[206]:


confusion_matrix(y, logistic_reg.predict(X))


# In[207]:


cm = confusion_matrix(y, logistic_reg.predict(X))

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=logistic_reg.classes_)
disp.plot()
plt.show()


# In[208]:


target_names = ['class 0', 'class 1']
print(classification_report(y, logistic_reg.predict(X), target_names=target_names))


# In[209]:


# Cross Validation


# In[210]:


for i in range(5, 11):
    kfold_validation = KFold(i, shuffle=True, random_state=0)
    results = cross_val_score(logistic_reg, X, y, cv=kfold_validation)
    print(f"For KFold = {i}:")
    print(f"{results}\n")


# <div style=" font-size:22px; line_height:160%">
# The result of cross validation shows that in the dataset, we do not have data that have a significant negative effect on score.

# In[211]:


df_sample = pd.DataFrame({
    "Age": [42],
#     "Experience": [16],
    "Income": [30],
    "ZIP_Code": [92037],
    "Family": [3],
    "CCAvg": [1.2],
    "Education": [3.0],
    "Mortgage": [0],
    "Securities_Account": [1],
    "CD_Account": [0],
    "Online": [1],
    "CreditCard": [1],
    "Personal_Loan": [1],
#     "ID": [5071]
})


# In[212]:


Df = pd.concat([df3, df_sample])
df_sample = Df.iloc[Df.shape[0]-1:]

df_sample


# In[213]:


X_train = df3.drop("Personal_Loan", axis=1)
y_train = df3["Personal_Loan"]#.values.reshape(-1, 1)
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[214]:


logistic_reg.fit(X_train, y_train)
y_pred_1 = logistic_reg.predict(X_test)


# In[215]:


print(y_pred_1)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# ComplementNB (Case 3)

# In[216]:


Counter(y)


# <div style=" font-size:22px; line_height:160%">
# This indicates that the target data is imbalanced. Because the number of data 0 is much more than the number of data 1.<br/>
# As a result, we use Naive Bayes Compliment.<br/>
# Also, for this reason, the accuracy metric is not a good measure and we should consider other metrics or the confusion matrix.

# In[217]:


X = df3.drop("Personal_Loan", axis=1)
y = df3["Personal_Loan"]#.values.reshape(-1, 1)


# In[218]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[219]:


clf_NB = ComplementNB()


# In[220]:


clf_NB.fit(X_train, y_train)


# In[221]:


y_pred_NB = clf_NB.predict(X_test)


# In[222]:


print(y_pred_NB)


# In[223]:


f1_score_NB_3 = metrics.f1_score(y_test, y_pred_NB)
print("Score (NB_3):", f1_score_NB_3)


# In[224]:


cm = confusion_matrix(y, clf_NB.predict(X))

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf.classes_)
disp.plot()
plt.show()


# In[225]:


target_names = ['class 0', 'class 1']
print(classification_report(y, clf_NB.predict(X), target_names=target_names))


# In[226]:


X_train = df3.drop("Personal_Loan", axis=1)
y_train = df3["Personal_Loan"]#.values.reshape(-1, 1)
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[227]:


clf_NB.fit(X_train, y_train)
y_pred_2 = clf_NB.predict(X_test)


# In[228]:


print(y_pred_2)


# <div style="font-family:B Nazanin; font-size:30px; color:#33FF8A; line-height: 130%">
# KNN Algorithm (Case 3)

# In[229]:


X = df3.drop("Personal_Loan", axis=1).values
y = df3["Personal_Loan"]#.values.reshape(-1, 1)


# In[230]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[231]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[232]:


K = 20
Acc = np.zeros(K)
print("Acc", Acc)

for i in range(1, K+1):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_scaling, y_train.ravel())
    y_pred = clf.predict(X_test_scaling)
    Acc[i-1] = metrics.f1_score(y_test, y_pred)
    
Acc


# In[233]:


print(np.max(Acc))
print(np.min(Acc))


# <div style=" font-size:22px; line_height:160%">
# Finding the appropriate k for recall score

# In[234]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.recall_score(y_train, y_train_new))
    test_acc.append(metrics.recall_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[235]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="recall",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[236]:


grid_kn.best_params_


# In[237]:


K = 1
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[238]:


f1_score_KNN_3_recall = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_3_recall):", f1_score_KNN_3_recall)


# In[239]:


X_train = df3.drop("Personal_Loan", axis=1).values
y_train = df3["Personal_Loan"]#.values.reshape(-1, 1)
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[240]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[241]:


clf.fit(X_train_scaling, y_train)
y_pred_3_1 = clf.predict(X_test_scaling)


# In[242]:


print(y_pred_3_1)


# <div style=" font-size:22px; line_height:160%">
# Finding the right k for precision

# In[243]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[244]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[245]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.precision_score(y_train, y_train_new))
    test_acc.append(metrics.precision_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[246]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="precision",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[247]:


grid_kn.best_params_


# In[248]:


K = 4
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[249]:


f1_score_KNN_3_precision = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_3_precision):", f1_score_KNN_3_precision)


# In[250]:


X_train = df3.drop("Personal_Loan", axis=1).values
y_train = df3["Personal_Loan"]#.values.reshape(-1, 1)
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[251]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[252]:


clf.fit(X_train_scaling, y_train)
y_pred_3_2 = clf.predict(X_test_scaling)


# In[253]:


print(y_pred_3_2)


# <div style=" font-size:22px; line_height:160%">
# Find the appropriate k for the score of f1

# In[254]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[255]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[256]:


training_acc = []
test_acc = []

# try KNN for different k nearest neighbor from 1 to 30
neighbors_settings = range(1, 31)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaling, y_train.ravel())
    y_train_new = knn.predict(X_train_scaling)
    y_test_new = knn.predict(X_test_scaling)
    training_acc.append(metrics.f1_score(y_train, y_train_new))
    test_acc.append(metrics.f1_score(y_test, y_test_new))
    
plt.plot(neighbors_settings, training_acc, label="Score of the training set")
plt.plot(neighbors_settings, test_acc, label="Score of the test set")
plt.ylabel("Score")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


# In[257]:


parameters = {"n_neighbors": range(1, 31)}
grid_kn = GridSearchCV(estimator=knn,  # Model
                      param_grid=parameters,  # Range of k
                      scoring="f1",  #Strategy to evaluate the performance
                      #of the cross-validated model on the test set
                      cv=5,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_kn.fit(X_train_scaling, y_train.ravel())


# In[258]:


grid_kn.best_params_


# In[259]:


K = 3
clf = KNeighborsClassifier(K)
clf.fit(X_train_scaling, y_train.ravel())
y_pred_KNN = clf.predict(X_test_scaling)


# In[260]:


f1_score_KNN_3_f1 = metrics.f1_score(y_test, y_pred_KNN)
print("Score (knn_3_f1):", f1_score_KNN_3_f1)


# In[261]:


X_train = df3.drop("Personal_Loan", axis=1).values
y_train = df3["Personal_Loan"]#.values.reshape(-1, 1)
X_test = df_sample.drop("Personal_Loan", axis=1).values


# In[262]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[263]:


clf.fit(X_train_scaling, y_train)
y_pred_3_3 = clf.predict(X_test_scaling)


# In[264]:


print(y_pred_3_3)


# <div style=" font-size:22px; line_height:160%">
# Conclusions:

# <div style=" font-size:22px; line_height:160%">
# Reminder:<br/>
# Mode number 1: We set the negative values of the experience to zero.<br/>
# Mode number 2: We removed negative experience values (about <a href="#inja" >1.04%</a> of all samples in the dataframe).<br/>
# Case number 3: We completely removed the experience column (feature) from the dataset due to high correlation with age.

# In[265]:


print("Score (lg_1):", f1_score_lg_1)
print("Score (lg_2):", f1_score_lg_2)
print("Score (lg_3):", f1_score_lg_3)


# In[266]:


print("Score (NB_1):", f1_score_NB_1)
print("Score (NB_2):", f1_score_NB_2)
print("Score (NB_3):", f1_score_NB_3)


# In[267]:


print("Score (knn_1_recall):", f1_score_KNN_1_recall)
print("Score (knn_2_recall):", f1_score_KNN_2_recall)
print("Score (knn_3_recall):", f1_score_KNN_3_recall)


# In[268]:


print("Score (knn_1_precision):", f1_score_KNN_1_precision)
print("Score (knn_2_precision):", f1_score_KNN_2_precision)
print("Score (knn_3_precision):", f1_score_KNN_3_precision)


# In[269]:


print("Score (knn_1_f1):", f1_score_KNN_1_f1)
print("Score (knn_2_f1):", f1_score_KNN_2_f1)
print("Score (knn_3_f1):", f1_score_KNN_3_f1)


# <div style=" font-size:22px; line_height:160%">
# According to the above values, it can be said that:<br/>
# - In general, removing the "experience" feature gives more accuracy to the models.<br/>
# - The Naive Bayes model does not have good accuracy on this dataset (of course, it should be noted that only the complementNB model was used among the Naive Biz models, because this model is used for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB" target="_blank">imbalanced</a> targets.<br/>
# - KNN models are more accurate than logistic regression models.

# In[ ]:





# In[ ]:





# In[270]:


df3


# In[291]:


X = df3.drop("Personal_Loan", axis=1).values
y = df3.Personal_Loan.values.reshape(-1, 1)


# In[292]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[293]:


dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# In[278]:


parameters = {"max_depth": range(1, 20), 
              "splitter": ["best", "random"]
             }

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid_dt = GridSearchCV(estimator=dt,  # Model
                      param_grid=parameters,
                      scoring="accuracy",  # Strategy to evaluate the performance of the cross-validated model on the test set
                                           # if it is a multiclass target, use f1_micro
                                           # f1 or roc_auc doesn't work with multiclass targets
                                           # f1_micro and accuracy were OK here.
                      cv=cv ,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_dt.fit(X_train, y_train.ravel())


# In[279]:


grid_dt.best_params_


# In[294]:


dt = DecisionTreeClassifier(max_depth=6, 
                            splitter='best', 
                            random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# In[295]:


f1_score_dt_3_f1 = metrics.f1_score(y_test, y_pred)
print("Score (dt_3_f1):", f1_score_dt_3_f1)


# In[281]:


cm = confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot()
plt.show()


# In[296]:


target_names = ["class '0'", "class '1'"]
print(classification_report(y_test, y_pred, target_names=target_names))


# In[283]:


df_sample = pd.DataFrame({
    "Age": [42],
#     "Experience": [16],
    "Income": [30],
    "ZIP_Code": [92037],
    "Family": [3],
    "CCAvg": [1.2],
    "Education": [3.0],
    "Mortgage": [0],
    "Securities_Account": [1],
    "CD_Account": [0],
    "Online": [1],
    "CreditCard": [1],
    "Personal_Loan": [1],
#     "ID": [5071]
})


# In[284]:


Df = pd.concat([df3, df_sample])
df_sample = Df.iloc[Df.shape[0]-1:]

df_sample


# In[300]:


X_train = df3.drop("Personal_Loan", axis=1).values
y_train = df3["Personal_Loan"].values.reshape(-1, 1)
X_test = df_sample.drop("Personal_Loan", axis=1)


# In[301]:


dt.fit(X_train, y_train)
y_pred_1 = dt.predict(X_test)


# In[302]:


print(y_pred_1)


# <div style=" font-size:22px; line_height:160%">
# As you can see, the accuracy of the tree model on imbalanced data is above 90%, while the accuracy of Logistic Regression, Naive Bayes, and KNN models is from 30% to 70%.

# In[ ]:





# In[ ]:





# <div dir="rtl" style="font-family:B Nazanin; font-size:22px; color:#33FF8A; line-height: 130%">
# 
