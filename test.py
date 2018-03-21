import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

def load_train_data(file_name):
    '''
    载入数据并进行预处理
    :param file_name:文件路径及文件名
    :return: X, Y [np.array]
    '''

    data = pd.read_csv(file_name, encoding='utf-8')
    data = preprocessing(data)
    
    return data
    
def preprocessing(data):
    '''
    对数据进行预处理
    :param data: DataFrame, 待处理的数据
    :return: DataFrame
    '''
    #查看缺失值数量：
    print(data.count())

    print("\nPre-processing...")

    #删除city特征，因为城市太多了额
    data = data.drop("City",axis = 1)

    #将DOB属性转换为Age，然后删除DOB
    data.DOB = list(map(lambda x: 118 - eval(x.split("-")[-1]), data.DOB))
    data.rename(columns={'DOB': 'Age'}, inplace=True)

    #EMI_Loan_Submitted_Missing = 1 如果 EMI_Loan_Submitted 丢失，否则为0
    data.EMI_Loan_Submitted = list(map(lambda x: 1 if str(x) == 'nan' else 0, data.EMI_Loan_Submitted))
    data.rename(columns={'EMI_Loan_Submitted': 'EMI_Loan_Submitted_Missing'}, inplace=True)

    #删除EmployerName字段,类别太多
    data = data.drop("Employer_Name", axis=1)

    #Existing_EMI如果丢失，用0填充: 共87020-86949个缺失
    data.Existing_EMI = data.Existing_EMI.fillna(0)

    #Interest_Rate_Missing = 1 如果 Interest_Rate 丢失，否则为0
    data.Interest_Rate = list(map(lambda x: 1 if str(x) == 'nan' else 0, data.Interest_Rate))
    data.rename(columns={'Interest_Rate': 'Interest_Rate_Missing'}, inplace=True)

    #Loan_Creation_Date删除
    data = data.drop("Lead_Creation_Date", axis=1)

    #Loan_Amount_Applied\Loan_Tenure_Applied
    data.Loan_Amount_Applied = data.Loan_Amount_Applied.fillna(data.Loan_Amount_Applied.median())
    data.Loan_Tenure_Applied = data.Loan_Tenure_Applied.fillna(data.Loan_Tenure_Applied.median())

    #Loan_Amount_Submitted_Missing = 1 Loan_Amount_Submitted 丢失，否则为0
    data.Loan_Amount_Submitted = list(map(lambda x: 1 if str(x) == 'nan' else 0, data.Loan_Amount_Submitted))
    data.rename(columns={'Loan_Amount_Submitted': 'Loan_Amount_Submitted_Missing'}, inplace=True)

    # Loan_Tenure_Submitted_Missing = 1 Loan_Tenure_Submitted 丢失，否则为0
    data.Loan_Tenure_Submitted = list(map(lambda x: 1 if str(x) == 'nan' else 0, data.Loan_Tenure_Submitted))
    data.rename(columns={'Loan_Tenure_Submitted': 'Loan_Tenure_Submitted_Missing'}, inplace=True)

    #删除LoggedIn、Salary_Account
    data = data.drop(["LoggedIn", "Salary_Account"], axis=1)

    #Processing_Fee_Missing = 1 Processing_Fee 丢失，否则为0
    data.Processing_Fee = list(map(lambda x: 1 if str(x) == 'nan' else 0, data.Processing_Fee))
    data.rename(columns={'Processing_Fee': 'Processing_Fee_Missing'}, inplace=True)

    #Source只保留top2，其他都成另外一类
    others = list(data.Source.value_counts().keys()[2:])
    data.Source = list(map(lambda x: x if x not in others else 'Others', data.Source))

    #Perform One-hot
    tem = pd.get_dummies(data.iloc[:,1:])
    data = pd.concat([data.iloc[:,0], tem], 1)

    data_c = list(data.columns)
    data_c[0] = "ID"
    data.columns = data_c

    return data

def modelfit(alg, dtrain, predictors, target, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    '''
    用于模型训练、测试的函数
    :param alg: 输入参数
    :param dtrain: 训练数据
    :param predictors: 用于预测的features
    :param useTrainCV: 是否使用交叉验证
    :param cv_folds: 交叉验证的folder数
    :param early_stopping_rounds: 最多运行多少轮
    :return: None
    '''

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold= cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True, show_stdv = True)
        alg.set_params(n_estimators = cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

train = load_train_data("data/Train_nyOWmfK.csv")
target = 'Disbursed'
IDcol = 'ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]


print("Train & Test ...")

#xgb1 = XGBClassifier(learning_rate =0.1,
#                     n_estimators=1000,
#                     max_depth=5,
#                     min_child_weight=1,
#                     gamma=0,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     objective= 'binary:logistic',
#                     nthread=4,
#                     scale_pos_weight=1,
#                     seed=27)
#
#modelfit(xgb1, train, predictors, target)

param_test2 = {'max_depth': [4,5,6],
               'min_child_weight':[4,5,6]}
               
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=5, silent = False,
                                                   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                                                   param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
                                                
gsearch1.fit(train[predictors],train[target])
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)