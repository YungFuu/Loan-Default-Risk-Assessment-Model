```python
import os,csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#from tqdm import tqdm
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

path = r'C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)'
df=pd.read_csv(path+os.sep+'credit_risk.csv')
```


```python
def draw_curve(fpr,tpr,roc_auc,save_name):
###make a plot of roc curve
    plt.figure(dpi=150)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(save_name)
    plt.legend(loc="lower right")
    plt.savefig(path+os.sep+save_name+'.jpg')
    plt.show()
    print('Figure was saved to ' + path)
```

# 简单逻辑回归


```python
LR = LogisticRegression()

###simple example: predictors include income and past_bad_credit
X=df[['income','past_bad_credit']]
y=df['default_label']

###run logistic regression
lr_model = LR.fit(X,y)

###another way to run logistic regression
lr_model1 = sm.Logit(y,sm.add_constant(X)).fit()
###get a summary result of lr
print(lr_model1.summary())
```

    Optimization terminated successfully.
             Current function value: 0.104542
             Iterations 9
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:          default_label   No. Observations:                13982
    Model:                          Logit   Df Residuals:                    13979
    Method:                           MLE   Df Model:                            2
    Date:                Sun, 07 Aug 2022   Pseudo R-squ.:                 0.01218
    Time:                        13:49:25   Log-Likelihood:                -1461.7
    converged:                       True   LL-Null:                       -1479.7
    Covariance Type:            nonrobust   LLR p-value:                 1.486e-08
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const              -5.3863      0.582     -9.252      0.000      -6.527      -4.245
    income           4.366e-05   7.94e-06      5.501      0.000    2.81e-05    5.92e-05
    past_bad_credit     1.2539      0.582      2.153      0.031       0.112       2.395
    ===================================================================================
    

#### 通过绘图，来观察模型拟合效果

####  True Positive Rate & False Positive Rate
真阳率（True Positive Rate, TPR）就是： 
含义是检测出来的真阳性样本（预测违约）数除以所有真实阳性样本数（真实违约）。  

假阳率（False Positive Rate, FPR）就是： 
含义是检测出来的假阳性样本（预计违约）数除以所有真实阴性（真实非违约）样本数。 

通俗点理解，即为：  

如果这个人确实得了病（违约），那么这个方法能检查出来的概率是多少呢？（预测违约的正确比例）  

如果这个人没有得病（正常还款），那么这个方法误诊其有病的概率是多少呢（假阳率）？（不给他贷款的错误概率） 

#### ROC（Receiver Operating Characteristic）
就是把假阳率当x轴，真阳率当y轴画一个二维平面直角坐标系。然后不断调整检测方法（或机器学习中的分类器）的阈值，即最终得分高于某个值就是阳性，反之就是阴性，得到不同的真阳率和假阳率数值，然后描点。就可以得到一条ROC曲线。 

需要注意的是，ROC曲线必定起于（0，0），止于（1，1）。  
因为，当全都判断为阴性(-)时，就是（0，0）；  
全部判断为阳性(+)时就是（1，1）。 

这两点间斜率为1的线段表示随机分类器（对真实的正负样本没有区分能力）。所以一般分类器需要在这条线上方。

#### AUC（Area Under Curve）
顾名思义，就是这条ROC曲线下方的面积了。越接近1表示分类器越好。 
但是，直接计算AUC很麻烦，但由于其跟Wilcoxon-Mann-Witney Test等价，所以可以用这个测试的方法来计算AUC。Wilcoxon-Mann-Witney Test指的是，任意给一个正类样本和一个负类样本，正类样本的score有多大的概率大于负类样本的score（score指分类器的打分）。


```python
###this is a two dimensional vector, prob d=0 and prob d=1, use the second one
predicted_prob = lr_model.predict_proba(X)
predicted_default_prob= predicted_prob[:,1]

###compute false positive rate and true positive rate using roc_curve function
fpr, tpr, _ = roc_curve(y, predicted_default_prob)
roc_auc = auc(fpr, tpr)
draw_curve(fpr,tpr,roc_auc,'2.1 Receiver operating characteristic example')
```


    
![output_8_0](https://user-images.githubusercontent.com/93023212/183279483-98648210-f88f-4857-be10-60a4bed973ff.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    


```python
#convert the gender, age and income
df['gender']=preprocessing.scale(df['gender'])
df['std_age']=preprocessing.scale(df['Age'])
df['std_income']=preprocessing.scale(df['income'])

#change the job_occupation to dummy
df['jo_0'] = pd.get_dummies(df['job_occupation'])[0]
df['jo_1'] = pd.get_dummies(df['job_occupation'])[1]

##change the edu to dummy
df['edu_0'] = pd.get_dummies(df['edu'])[0]
df['edu_1'] = pd.get_dummies(df['edu'])[1]
df['edu_2'] = pd.get_dummies(df['edu'])[2]
df['edu_3'] = pd.get_dummies(df['edu'])[3]
```

#### 使用遍历手段获得模型效果最好的feature组合


```python
#variables that we have tried
#df['dummy_edu']=list(map(lambda x: np.log(x),df['edu']))
#df['gender']=list(map(lambda x: np.log(x),df['gender']))
#df['ln_income']=list(map(lambda x: np.log(x),df['income']))
#df['std_ln_age']=preprocessing.scale(list(map(lambda x: np.log(x),df['Age'])))
#df['std_ln_income']=preprocessing.scale(list(map(lambda x: np.log(x),df['income'])))
#df['Age'] = df['Age']//10
#df['std_edu']=preprocessing.scale(df['edu'])

#Use Exhaustive method to try every combination,but at last, we find the best combination is the pool list now
#使用遍历手段，可以得到最佳的feature 组合，但变量较多的情况下，最好分析变量间相关性以后再做选择
LR = LogisticRegression(penalty="l1",solver= 'liblinear',class_weight='balanced',tol=0.008,max_iter=100000)
df2=pd.DataFrame()
cbna_list=[] #save variables
auc_list=[] #save auc
variables=[] #save number of variables

for i in tqdm(range(7,len(pool)+1)): #Use up to len（pool） variables
    for cbna in itertools.combinations(pool, i):
        
        X=df[list(cbna)]
        y=df['default_label']
        x_smote, y_smote = smote.fit_resample(X, y)
        lr_model = LR.fit(x_smote,y_smote)
        predicted_prob = lr_model.predict_proba(x_smote)
        predicted_default_prob= predicted_prob[:,1]
        fpr, tpr, _ = roc_curve(y_smote, predicted_default_prob)
        roc_auc = auc(fpr, tpr)
            
        #save results
        cbna_list.append(list(cbna))
        variables.append(len(list(cbna)))
        auc_list.append(roc_auc)


df2['Varibles']=cbna_list
df2['No. of variables used'] = variables
df2['auc value'] = auc_list
```


```python
pool=['gender',
 'housing',
 'income',
 'std_age',
 'past_bad_credit',
 'married',
 'jo_0',
 'edu_0',
 'edu_1',
 'edu_2']
```

#### 相关性分析


```python
import seaborn as sns
mean_corr = df[pool].corr()
plt.figure(figsize=(14,8))
sns.heatmap(mean_corr,annot=True)
plt.show()
```


    
![output_14_0](https://user-images.githubusercontent.com/93023212/183279492-3f881d2d-3925-4325-8310-b72845e1eb73.png)
    


##### 对Feature进行调整和筛选后，模型效果得到进一步提升


```python
pool=['gender',
 'housing',
 'income',
 'std_age',
 'past_bad_credit',
 'married',
 'jo_0',
 'jo_1',
 'edu_0',
 'edu_1',
 'edu_2',
 'edu_3']

LR = LogisticRegression(penalty="l1",solver= 'liblinear',class_weight={1:0.8,0:0.2},
                        tol=0.008,max_iter=100000)

X = df[pool]
y=df['default_label']
lr_model = LR.fit(X,y)
lr_model1 = sm.Logit(y,sm.add_constant(X)).fit()
predicted_prob = lr_model.predict_proba(X)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y, predicted_default_prob)
roc_auc = auc(fpr, tpr)
print(lr_model1.summary())
print('the best combination: ', list(X.columns))
print('used variables: ' , len(X.columns))
print('the auc value: ' , roc_auc)

draw_curve(fpr,tpr,roc_auc,'2.2 Full Logistic Model')

```

    Optimization terminated successfully.
             Current function value: 0.100597
             Iterations 9
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:          default_label   No. Observations:                13982
    Model:                          Logit   Df Residuals:                    13969
    Method:                           MLE   Df Model:                           12
    Date:                Sun, 07 Aug 2022   Pseudo R-squ.:                 0.04946
    Time:                        14:12:21   Log-Likelihood:                -1406.6
    converged:                       True   LL-Null:                       -1479.7
    Covariance Type:            nonrobust   LLR p-value:                 3.093e-25
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const              -6.6571      0.814     -8.183      0.000      -8.252      -5.063
    gender              0.0522      0.059      0.887      0.375      -0.063       0.168
    housing            -1.0852      0.141     -7.699      0.000      -1.361      -0.809
    income           7.392e-05   1.02e-05      7.263      0.000     5.4e-05    9.39e-05
    std_age             0.0750      0.070      1.068      0.285      -0.063       0.212
    past_bad_credit     1.2635      0.583      2.166      0.030       0.120       2.407
    married             0.0048      0.120      0.040      0.968      -0.230       0.239
    jo_0                0.9016      0.466      1.937      0.053      -0.011       1.814
    jo_1                0.4484      0.483      0.928      0.353      -0.498       1.395
    edu_0               1.2317      0.379      3.253      0.001       0.490       1.974
    edu_1               1.0006      0.351      2.850      0.004       0.313       1.689
    edu_2               0.7626      0.331      2.305      0.021       0.114       1.411
    edu_3               0.3237      0.333      0.973      0.330      -0.328       0.976
    ===================================================================================
    the best combination:  ['gender', 'housing', 'income', 'std_age', 'past_bad_credit', 'married', 'jo_0', 'jo_1', 'edu_0', 'edu_1', 'edu_2', 'edu_3']
    used variables:  12
    the auc value:  0.6822061586212529
    


    
![output_16_1](https://user-images.githubusercontent.com/93023212/183279501-524d70b6-a27b-4b63-89e9-4c855574146f.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

#### 样本类别不平衡问题
类别不平衡问题，顾名思义，即数据集中存在某一类样本，其数量远多于或远少于其他类样本，从而导致一些机器学习模型失效的问题。例如逻辑回归即不适合处理类别不平衡问题，例如逻辑回归在欺诈检测问题中，因为绝大多数样本都为正常样本，欺诈样本很少，逻辑回归算法会倾向于把大多数样本判定为正常样本，这样能达到很高的准确率，但是达不到很高的召回率。


```python
from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
```

    Counter({0: 13674, 1: 308})
    

##### 
通过imblearn的SMOTE函数，可以将样本不平衡的问题解决，将平衡后的样本放入模型再做测试，会发现拟合效果得到了大幅提升


```python
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
# 可通过radio参数指定对应类别要生成的数据的数量
#smo = SMOTE(ratio={1: 300 },random_state=42)
X = df[pool]
y=df['default_label']
x_smote, y_smote = smote.fit_resample(X, y)
```


```python
print(Counter(y_smote)) #现在获得了类别1：1的样本
```

    Counter({0: 13674, 1: 13674})
    


```python
LR = LogisticRegression(penalty="l1",solver=
'liblinear',class_weight='balanced',tol=0.008,max_iter=100000)
lr_model = LR.fit(x_smote,y_smote)
lr_model1 = sm.Logit(y_smote,sm.add_constant(x_smote)).fit()
predicted_prob = lr_model.predict_proba(x_smote)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y_smote, predicted_default_prob)
roc_auc = auc(fpr, tpr)
print(lr_model1.summary())
print('the best combination: ', list(X.columns))
print('used variables: ' , len(X.columns))
print('the auc value: ' , roc_auc)
draw_curve(fpr,tpr,roc_auc,'2.2 Full Logistic Model')
```

    Optimization terminated successfully.
             Current function value: 0.332657
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:          default_label   No. Observations:                27348
    Model:                          Logit   Df Residuals:                    27335
    Method:                           MLE   Df Model:                           12
    Date:                Sun, 07 Aug 2022   Pseudo R-squ.:                  0.5201
    Time:                        14:30:02   Log-Likelihood:                -9097.5
    converged:                       True   LL-Null:                       -18956.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const               4.9757      0.137     36.263      0.000       4.707       5.245
    gender             -0.0556      0.020     -2.723      0.006      -0.096      -0.016
    housing            -2.4900      0.050    -49.633      0.000      -2.588      -2.392
    income           4.751e-05   3.44e-06     13.817      0.000    4.08e-05    5.42e-05
    std_age             0.2766      0.024     11.535      0.000       0.230       0.324
    past_bad_credit     0.4397      0.112      3.933      0.000       0.221       0.659
    married            -1.1174      0.041    -27.345      0.000      -1.198      -1.037
    jo_0               -2.0584      0.061    -33.933      0.000      -2.177      -1.940
    jo_1               -3.7146      0.085    -43.950      0.000      -3.880      -3.549
    edu_0              -2.7959      0.071    -39.390      0.000      -2.935      -2.657
    edu_1              -3.3709      0.066    -51.452      0.000      -3.499      -3.243
    edu_2              -3.1943      0.059    -54.122      0.000      -3.310      -3.079
    edu_3              -3.8529      0.071    -54.159      0.000      -3.992      -3.713
    ===================================================================================
    the best combination:  ['gender', 'housing', 'income', 'std_age', 'past_bad_credit', 'married', 'jo_0', 'jo_1', 'edu_0', 'edu_1', 'edu_2', 'edu_3']
    used variables:  12
    the auc value:  0.930610227682279
    


    
![output_22_1](https://user-images.githubusercontent.com/93023212/183279506-f10a1777-057d-4701-bb31-4272963de1af.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

# 划分训练集和测试集


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_smote,
                                                    y_smote,
                                                    train_size=15000,
                                                    random_state=1)

lr_model = LR.fit(X_train,y_train)
predicted_prob = lr_model.predict_proba(X_test)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y_test, predicted_default_prob)
roc_auc = auc(fpr, tpr)
draw_curve(fpr,tpr,roc_auc,'2.4 Out-of-Sample Test')
```


    
![output_24_0](https://user-images.githubusercontent.com/93023212/183279510-4d76ffda-2817-49ce-8558-b9e52950be97.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

# SVM 支持向量机

##### '''
SVC(  
    C=1.0, #惩罚系数，即正则化，  
            #当C越大时，分类器的准确性越高，所以容错率越低，泛化能力就变差。  
            #当C越小时，分类器的准确性降低，但容错率增大，泛化能力越强  
    kernel='rbf', #选择使用的函数  
    degree=3,  
    gamma='scale',  
    coef0=0.0,  
    shrinking=True,  
    probability=False,  
    tol=0.001,  
    cache_size=200,  
    class_weight=None, #调整权重  
    verbose=False,  
    max_iter=-1, #最大迭代次数  
    decision_function_shape='ovr', #ovr 一对多 或 ovo 一对一  
    break_ties=False,  
    random_state=None,  
)  
'''


```python
from sklearn.svm import SVC
regressor = SVC(C=1, cache_size=1024, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


regressor.fit(X_train, y_train)
predicted_prob = regressor.predict_proba(X_test)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y_test, predicted_default_prob)
roc_auc = auc(fpr, tpr)
draw_curve(fpr,tpr,roc_auc,'SVM')
```


    
![output_27_0](https://user-images.githubusercontent.com/93023212/183279513-430cacc5-266c-443d-803a-191dff31f4fb.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

# Decision Tree 决策树

##### 
clf=tree.DecisionTreeClassifier(  
    criterion='gini',  #分类原理，熵，基尼或log_loss  
    splitter='best', #输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查  看），输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这也是防止过拟合的一种方式。
    max_depth=5,      #数的最大深度，若None 节点可以无限延申，直到全部分净,一般从3开始尝试  
    min_samples_split=2, #能建立node的最小样本量  
    min_samples_leaf=5,  #一个叶片上所需要有的最小样本量，一般从5开始尝试，若类别不多，可以设为1  
    min_weight_fraction_leaf=0.0,     
    max_features=None,  # 一个节点考虑的最大特征数  
    random_state=None,  #用来设置分枝中的随机模式的参数，  
    max_leaf_nodes=20,  #最大节点数（防止过拟合）  
    min_impurity_decrease=0.0,  #  
    class_weight='balanced',  #对样本量权重进行调整  
    ccp_alpha=0.0)#实例化  


```python
from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion='gini',  #分类原理，熵，基尼或log_loss
    splitter='best', #输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看），输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这也是防止过拟合的一种方式。
    max_depth=5,      #数的最大深度，若None 节点可以无限延申，直到全部分净,一般从3开始尝试
    min_samples_split=2, #能建立node的最小样本量
    min_samples_leaf=5,  #一个叶片上所需要有的最小样本量，一般从5开始尝试，若类别不多，可以设为1
    min_weight_fraction_leaf=0.0,   
    max_features=None,  # 一个节点考虑的最大特征数
    random_state=None,  #用来设置分枝中的随机模式的参数，
    max_leaf_nodes=20,  #最大节点数（防止过拟合）
    min_impurity_decrease=0.0,  #
    class_weight='balanced',  #对样本量权重进行调整
    ccp_alpha=0.0)#实例化

clf.fit(X_train, y_train)
predicted_prob = clf.predict_proba(X_test)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y_test, predicted_default_prob)
roc_auc = auc(fpr, tpr)
draw_curve(fpr,tpr,roc_auc,'Decision Tree')
```


    
![output_30_0](https://user-images.githubusercontent.com/93023212/183279516-1b675ead-6a97-4400-962d-76974ab27670.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

##### 决策树实现节点可视化


```python
import graphviz
dot_data=tree.export_graphviz(clf,
                              feature_names=pool,
                              filled=True,#填充颜色，颜色越深，不纯度越低
                              rounded=True#框的形状
                              )
graph=graphviz.Source(dot_data)
graph

```




    
![image](https://user-images.githubusercontent.com/93023212/183279545-23e92aa2-0e3b-45c2-83cc-8c5982c606f9.png)
    



##### 通过循环语句等来调参，确定深度最大为何时最优


```python
from tqdm import tqdm
AUCs=[]
scores=[]
for i in tqdm(range(1,100)):
    clf=tree.DecisionTreeClassifier(criterion='gini',  #分类原理，熵，基尼或log_loss
        splitter='best', #输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看），输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这也是防止过拟合的一种方式。
        max_depth=i,      #数的最大深度，若None 节点可以无限延申，直到全部分净,一般从3开始尝试
        min_samples_split=2, #能建立node的最小样本量
        min_samples_leaf=5,  #一个叶片上所需要有的最小样本量，一般从5开始尝试，若类别不多，可以设为1
        min_weight_fraction_leaf=0.0,   
        max_features=None,  # 一个节点考虑的最大特征数
        random_state=None,  #用来设置分枝中的随机模式的参数，
        max_leaf_nodes=20,  #最大节点数（防止过拟合）
        min_impurity_decrease=0.0,  #
        class_weight='balanced',  #对样本量权重进行调整
        ccp_alpha=0.0)#实例化
    
    clf.fit(X_train, y_train)
    predicted_prob = clf.predict_proba(X_test)
    predicted_default_prob= predicted_prob[:,1]
    fpr, tpr, _ = roc_curve(y_test, predicted_default_prob)
    roc_auc = auc(fpr, tpr)
    AUCs.append(roc_auc)
    score = clf.score(X,y)
    scores.append(score)

plt.plot(range(1,100),AUCs,color="red",label="max_depth")
plt.legend()
plt.show()
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 99/99 [00:02<00:00, 43.17it/s]
    


    
![output_34_1](https://user-images.githubusercontent.com/93023212/183279553-02d9119a-0dfe-4b36-9e97-3acca787fd60.png)
    


# GBDT 梯度提升决策树


```python
from sklearn import ensemble
clf =ensemble.GradientBoostingClassifier()
#clf = ensemble.GradientBoostingRegressor()


clf = ensemble.GradientBoostingClassifier(
    loss='log_loss', #GBDT的损失函数，分类模型和回归模型的损失函数不一样
    learning_rate=0.01, #学习率
    n_estimators=1000, #学习器最大迭代次数
    subsample=1.0,#子采样，随机森林是有放回的采样，这里是不放回的采样
    criterion='friedman_mse',
    min_samples_split=2,  #限制子树继续划分的条件，如果某节点样本数量少于它，就不再继续划分
    min_samples_leaf=1, #限制叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会跟节点一起去掉
    min_weight_fraction_leaf=0.0,
    max_depth=4, #决策树的最大深度
    min_impurity_decrease=0.0,
    init=None,
    random_state=None,
    max_features=None,  #划分时考虑的最大特征数
    verbose=0,
    max_leaf_nodes=3, #最大叶子节点数，防止过拟合
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001,
    ccp_alpha=0.0,
    )
```


```python
gbdt_model = clf.fit(X_train,y_train)   

predicted_prob = gbdt_model.predict_proba(X_test)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y_test, predicted_default_prob)
roc_auc = auc(fpr, tpr)
draw_curve(fpr,tpr,roc_auc,'GBDT')
```


    
![output_37_0](https://user-images.githubusercontent.com/93023212/183279555-d8a20890-b416-4b67-ae1b-2c0dea356d6a.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

# XGboost


```python
import xgboost as xgb

model = xgb.XGBClassifier(max_depth=3, n_estimators=200)
model.fit(X_train, y_train)  
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
              importance_type=None, interaction_constraints=&#x27;&#x27;,
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=1,
              missing=nan, monotone_constraints=&#x27;()&#x27;, n_estimators=200,
              n_jobs=0, num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0,
              reg_alpha=0, reg_lambda=1, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
              importance_type=None, interaction_constraints=&#x27;&#x27;,
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=1,
              missing=nan, monotone_constraints=&#x27;()&#x27;, n_estimators=200,
              n_jobs=0, num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0,
              reg_alpha=0, reg_lambda=1, ...)</pre></div></div></div></div></div>




```python
predicted_prob = model.predict_proba(X_test)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y_test, predicted_default_prob)
roc_auc = auc(fpr, tpr)
draw_curve(fpr,tpr,roc_auc,'XGboost')
```


    
![output_40_0](https://user-images.githubusercontent.com/93023212/183279557-591e23fa-548b-4268-b36b-01594aa4d635.png)
    


    Figure was saved to C:\Users\pc\Desktop\HKU\Course Learning\MFIN7034\MFIN7034 Problem Set2 (Fu Yangyang, 3035882158)
    

# 总结
将几种常见的贷款违约风险分析模型的预测效果进行对比：  
LR: AUC=0.93，速度极快  
SVM：AUC=0.82，速度较慢  
DT：AUC=0.91，速度极快    
GBDT：AUC=0.96，速度较快  
XGBoost: AUC:0.99 速度较快  


```python

```
