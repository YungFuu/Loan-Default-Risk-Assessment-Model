# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:23:01 2023

@author: pc
"""

import pandas as pd
import numpy as np
import gc,os
from sklearn.preprocessing import  OneHotEncoder

path = r'C:\Users\pc\Desktop\data'


data_pre_application = pd.read_csv(path + os.sep + 'previous_application.csv')


enc = OneHotEncoder()

#异常值替换为缺失
data_pre_application['DAYS_FIRST_DRAWING'].replace(365243,np.nan,inplace= True)
data_pre_application['DAYS_FIRST_DUE'].replace(365243, np.nan,inplace=True)
data_pre_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243,np.nan,inplace= True)
data_pre_application['DAYS_LAST_DUE'].replace(365243, np.nan,inplace=True)
data_pre_application['DAYS_TERMINATION'].replace(365243, np.nan,inplace=True)

#字符型特征高散化?????
enc.fit(data_pre_application)
X_encoded  = enc.transform(data_pre_application)

data_pre_application,new_columns_pre_application = enc.fit(data_pre_application)

#衍生比例类特征
data_pre_application["APP_CREDIT_PREC"] = data_pre_application['AMT_APPLICATION'] / data_pre_application['AMT_CREDIT']

#衍生聚合类特征

num_aggregations = {
    'AMT_ANNUITY':['min','max','mean'],
    'AMT_APPLICATION':['min','max','mean'],
    'AMT_CREDIT':['min','max','mean'],
    'AMT_CREDIT_PERC':['min','max','mean','var'],
    'AMT_DOWN_PAYMENT':['min','max','mean'],
    'AMT_GOODS_PRICE':['min','max','mean'],
    'HOUR_APPR_PROCESS_START':['min','max','mean'],
    'RATE_DOWN_PAYMENT':['min','max','mean'],
    'DAYS_DECISION':['min','max','mean'],
    'CNT_PAYMENT':['sum','mean']
    }

cat_aggregations = {}

for cat in new_columns_pre_application:
    cat_aggregations['cat'] = ['mean']


#历史申请特征
prev_agg = data_pre_application.groupby('SK_ID_CURR').agg({**num_aggregations,**cat_aggregations})
prev_agg.columns = pd.Index(['PREV_'+column[0]+'_'+ column[1].upper() for column in prev_agg.columns.tolist()])

#历史通过申请特征
approved = data_pre_application[data_pre_application['NAME_CONTRACT_STATUS_Approved'] == 1]
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
approved_agg.columns =pd.Index(['APPROVED_' + column[0]+'_'+column[1].upper() for column in approved_agg.columns.tolist()])
prev_agg = prev_agg.join(approved_agg,how = 'left',on = 'SK_ID_CURR')
del approved,approved_agg

#历史申请拒绝特征
refused = data_pre_application[data_pre_application['NAME_CONTRACT_STATUS_REFUSED'] == 1]
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
refused_agg.columns =pd.Index(['APPROVED_' + column[0]+'_'+column[1].upper() for column in refused_agg.columns.tolist()])
prev_agg = prev_agg.join(refused_agg,how = 'left',on = 'SK_ID_CURR')
del refused,refused_agg,data_pre_application

gc.collect()






#%% 探索性数据分析
def EDA_series(data):
    '''
    这个函数是一个用于探索性数据分析（Exploratory Data Analysis, EDA）的函数。
    它可以对输入的数据(某一特征)进行基本的统计分析和探索性分析，生成包括数据总数、缺失值数量、缺失值比例、唯一值数量、类型、最大值、最小值、均值、标准差、异常值数量和异常值比例等统计量的字典。
    这些统计量可以帮助数据分析人员快速了解数据集的基本情况，进而进行下一步的数据清洗、特征选择、特征工程等操作。

    Parameters
    ----------
    data : DataFrame
        series.原始数据

    Returns
    -------
    series. 原始数据EDAjieguo 

    '''

    result = {}
    result['count'] = len(data)
    result['missing_count'] = data.isnull().sum()
    result['missing_rate'] = "{:.2%}".format(result['missing_count'] / result['count'])
    result['count_unique'] = len(data.value_counts(normalize = True))
    data.dropna(inplace = True)
    if data.dtype == 'object':
        result['type'] = 'categorical'
    else:
        result['type'] = 'numeric'
        result['max'] = data.max()
        result['min'] = data.min()
        result['mean'] = round(data.mean(),2)
        result['std'] = round(data.std(),2)
        zscore = (data-data.mean()) / data.std()
        result['outlier_count'] = (zscore.abs() > 6).sum()
        result['outlier_rate'] = "{:.2%}".format(result['outlier_count'] / result['count'])
    if result['count_unique'] <= 2:
        result['type'] = 'binary'
        
        
        
    result = pd.Series(result)
    result = result.reindex(index = ['type','count','count_unique','missing_count','missing_rate','min','max','mean','std','outlier_count','outlier_rate'])
    return result
        
        
        
    
        

def EDA_df(data):
    '''
    探索性数据分析

    Parameters
    ----------
    data : DataFrame
        需要进行探索性分析的数据。

    Returns
    -------
    result : DataFrame
        返回一个包含数据类型、数据数量、唯一值数量、缺失值数量、缺失率、最大值、最小值、均值、标准差、异常值数量和异常值比例等统计量的DataFrame类型数据。

    '''
    
    
    result  = []
    for column in data.columns.tolist():
        tmp = EDA_series(data[column])
        tmp.name = column
        result.append(tmp)
    result = pd.concat(result,axis=1).T
    columns_result = ['type','count','count_unique','max','min','mean','std',
                      'missing_count','missing_rate','outlier_count','outlier_rate']
    result = result[columns_result]
    return result

    
        
result = EDA_df(data_pre_application)
    
    



def discretize(data,columns_continous,quantiles):
    '''
    这个函数用于将连续型数据转化为离散型数据，将数据划分为quantiles个区间，每个区间用区间的左右端点表示，例如[0,1)表示[0,1)区间内的所有数。
    具体来说，对于每个连续型特征，将其分为quantiles-1个区间，每个区间用区间的左右端点表示，最后一个区间用区间的左端点和最大值表示。
    空值将被标记为"nan"。

    Parameters
    ----------
    data : DataFrame
        需要被处理的数据.
    columns_continous : list
        需要被转化为离散型数据的特征.
    quantiles : int
        每个连续型特征需要被分为几个区间.

    Returns
    -------
    data_bin : DataFrame
        DESCRIPTION.

    '''
    data_bin = data.copy()
    columns_cate = [column for column in data_bin.columns if column not in columns_continous]
    for column in columns_continous:
        X = data_bin[column].copy()
        for i in range(len(quantiles)-1):
            left = X.quantile(quantiles[i])
            right = X.quantile(quantiles[i+1])
            if i < len(quantiles)-2:
                group = '[' + str(left)+','+str(right)+')'
                data_bin[column].iloc[np.where((X>=left) & (X<right))] = group
            if i == len(quantiles)-2:
                group = '[' + str(left)+','+str(right)+')'
                data_bin[column].iloc[np.where((X>=left) & (X<=right))] = group
        data_bin[column].fillna('nan',inplace = True)
    for column in columns_cate:
        data_bin[column] = data_bin[column].astype(str)
    return data_bin

def woe_iv_calc(data_bin,y):
    '''
    计算WOE和IV函数

    Parameters
    ----------
    data_bin : DataFrame
        经过分箱后的数据.
    y : Series
        目标变量，值为0或1.

    Returns
    -------
    data_woe : DataFrame
        WOE映射后的数据.
    map_woe : Dict
        Key为变量名，value 为每个箱对应的WOE值.
    map_iv : Dict
        Key为变量名，value 为 IV 值.

    '''
    data_woe = data_bin.copy()
    map_woe = {}
    map_iv = {}
    
    for column in data_woe.columns:
        cross = pd.crosstab(data_woe[column],y)
        cross[cross ==0] = 1 #解决分母为0的问题
        cross = cross/cross.sum(axis = 0)
        woe = np.log(cross[0]/cross[1])
        iv = ((cross[0] - cross[1])*np.log(cross[0] - cross[1])).sum()
        map_woe[column] = dict(woe)
        map_iv[column] = iv
        data_woe[column] = data_woe[column].map(dict(woe))
    return data_woe,map_woe,map_iv


X_columns = data_lr.columns[2:]
Y_columns = 'TARGET'
columns_continous = eda_stat[eda_stat['count_unique']>10].index.tolist()
columns_continous = [column for column in columns_continous if column != 'SK_ID_CURR']
quantiles = [0.1 * i for i in range(11)]
data_bin = discretize(data_lr[X_columns],columns_continous,quantiles)
data_woe,map_woe,map_iv = woe_iv_calc(data_bin,data_lr[Y_columns])        
        
        
        
        
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#计算相关性
data_cor = data_lr[columns_select].corr().abs()
data_cor_lower = pd.DataFrame(np.tril(data_cor),index = data_cor.index,columns = data_cor.columns)
        
columns_drop = []

for column in data_cor_lower:
    data_cor_select  = data_cor_select = data_cor_lower[(data_cor_lower[column]>0.8) & (data_cor_lower[column]<1)]
    if len(data_cor_select)>0:
        data_cor_select = pd.DataFrame(data=data_cor_select.columns.tolist() 
                                       + data_cor_select.index.tolist(),
                                       columns = ['column_name']
                                       )
        data_cor_select['IV'] = data_cor_select['column_name'].map(map_iv)
        data_cor_select = data_cor_select.sort_values(by ='IV',ascending = False)
        columns_drop = columns_drop + data_cor_select['column_name'].to_list()[1:]
        
columns_select = [column for column in columns_select if column not in columns_drop]
data_lr = data_lr[['SK_ID_CURR','TARGET']+columns_select]

#计算多重共线性
data_vif = data_lr.iloc[:, 2:].copy()
data_vif = sm.add_constant(data_vif)
data_vif = data_vif.replace([np.nan, np.inf], -9999)

vif_select = pd.DataFrame(data=data_vif.columns, columns=['column_name'])
vif_select['VIF'] = [variance_inflation_factor(data_vif.values, i) for i in range(data_vif.shape[1])]

columns_select = data_lr.iloc[:, 2:].columns[vif_select['VIF'] < 10].tolist()
data_lr = data_lr[['SK_ID_CURR', 'TARGET'] + columns_select]
        
        
            
       
        
def woe_plot(map_woe,close = True,show_last = True):
    '''
    生成各个特征的WOE值分布图

    Parameters
    ----------
    map_woe : Dict
        Key为变量名，value为每箱对应的WOE值，建议每箱预先排序方便观察单调性
    close : Bool, optional
        是否打印WOE值分布图. The default is True.
    show_last : Bool, optional
        是否只保留最后一个变量的WOE值分布图. The default is True.

    Returns
    -------
    result : Dict
        Key为变量名，value为每个变量的WOE值分布图.

    '''
    import matplotlib as plt
    result = {}
    for i,feature in enumerate(map_woe):
        data = pd.Series(map_woe[feature])
        data.index.name = ''
        data.name = ''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data.plot(kind = 'bar',ax =ax)
        ax.set_xlabel('变量分箱')
        ax.set_ylabel('WOE值')
        ax.set_title('%s' %feature)
        result[feature] = fig
        if close and show_last and i<len(map_woe)-1:
            plt.close('all')
    return result
        
        
        
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression

data_lr[columns_select] = data_woe[columns_select]
X_columns = data_lr.columns[2:]
Y_columns = 'TARGET'
X_train,X_test,y_train,y_test = train_test_split(data_lr[X_columns],data_lr[Y_columns],test_size = 0.3,random_state= 0)
tuner_parameters = [{'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10]}]
clf = GridSearchCV(LogisticRegression(), tuner_parameters,cv =5,scoring = 'roc_auc')
clf.fit(X_train,y_train)
clf.best_params_
lr = LogisticRegression(penalty='l2',C = 0.1)
lr_clf = lr.fit(X_train,y_train)
        
        
#%%        
        
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve,auc,roc_auc_socre

X_columns = data_xgb.columns[2:]
Y_columns = ['TARGET']

X_train,X_test,y_train,y_test = train_test_split(data_xgb[X_columns],
                                                 data_xgb[Y_columns],
                                                 test_size = 0.3,
                                                 random_state=0
                                                 )

X_matrix_train = X_train.as_matrix(columns = None)
Y_matrix_train = Y_train.as_matrix(columns = None)
X_matrix_test = X_test.as_matrix(columns = None)
Y_matrix_test = Y_test.as_matrix(columns = None)

















        
        
        
        
        
        
        