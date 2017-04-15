# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:31:18 2016

@author: Shreyas Rewagad
"""

from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

HouseTrain = pd.read_csv("./train.csv", index_col = 0, parse_dates=True, header = 0)
#HouseTrain = pd.read_csv("C:/Users/Shreyas Rewagad/Desktop/Data Mining/Kaggle/train.csv", index_col = 0, parse_dates=True, header = 0)

Sp = HouseTrain['SalePrice']
HouseTrain = HouseTrain.drop('SalePrice',1)

HouseTest = pd.read_csv("./test.csv", index_col = 0, parse_dates=True, header = 0)
#HouseTest = pd.read_csv("C:/Users/Shreyas Rewagad/Desktop/Data Mining/Kaggle/test.csv", index_col = 0, parse_dates=True, header = 0)

House = HouseTrain.append(HouseTest, ignore_index=False,verify_integrity=True)

###########################################################################################

withNone = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
for i in withNone:
    House[i] = np.where(House[i].isnull(), "NONE",House[i])

cat = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType'
]
for i in cat:
    House[i] = np.where(House[i].isnull(),House[i].mode(),House[i])
    
House['MasVnrArea'] = np.where(House['MasVnrArea'].isnull(), House['MasVnrType'] == "NONE",House['MasVnrArea'])    
    
House['LotFrontage'] = np.where(House['LotFrontage'].isnull(), House['LotFrontage'].mean(),House['LotFrontage'])

House['BsmtFinSF1'] = np.where(House['BsmtFinType1'] == "NONE",0,House['BsmtFinSF1'])
House['BsmtFinSF2'] = np.where(House['BsmtFinType2'] == "NONE",0,House['BsmtFinSF2'])
House['BsmtUnfSF'] = np.where(House['BsmtQual'] == "NONE",0,House['BsmtUnfSF'])
House['BsmtFullBath'] = np.where(House['BsmtQual'] == "NONE" ,0,House['BsmtFullBath'])
House['BsmtHalfBath'] = np.where(House['BsmtQual'] == "NONE" ,0,House['BsmtHalfBath'])
House['TotalBsmtSF'] = np.where(House['BsmtQual'] == "NONE" ,0,House['TotalBsmtSF'])

House['GarageYrBlt'] = np.where(House['GarageFinish']== 'NONE', House['YearBuilt'],House['GarageYrBlt'])
House['GarageCars'] = np.where(House['GarageFinish'] == 'NONE', 0,House['GarageCars'])
House['GarageArea'] = np.where(House['GarageFinish'] == 'NONE', 0,House['GarageArea'])

#########################################################################################################
##After Boruto Feature Selection
#print(House.shape)
House = House[['MSSubClass','MSZoning','LotFrontage','LotArea','Alley','LotShape','LandContour','Neighborhood','Condition1','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','Foundation','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','GrLivArea','BsmtFullBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','ScreenPorch','Fence','SaleCondition']]
#print(House.shape)
df = pd.DataFrame(index = House.index)
cols = House.columns
num_cols = House._get_numeric_data().columns
cat_cols = cols.difference(num_cols)
for c in cat_cols:
    df = pd.concat([df, pd.get_dummies(House[c], prefix=c, drop_first = False)], axis=1)
for c in num_cols:
    df = pd.concat([df, House[c]], axis=1)
#

Train = df.head(1460)
Train['SalePrice'] = Sp
#Train.to_csv("C:/Users/Shreyas Rewagad/Desktop/TrainData.csv", sep='\t', encoding='utf-8')
#
Test = df.tail(1459)
#Test.to_csv("C:/Users/Shreyas Rewagad/Desktop/TestData.csv", sep=',', encoding='utf-8')
###########################################################################

###########################################################################

Feat = {}

for i in range(100):
    Train_Sample = Train.sample(frac=random.uniform(0.3, 1), replace=False)
    Y = Train_Sample['SalePrice']
    X = Train_Sample.drop('SalePrice',1)
    names = list(X.columns.values)
    rf = RandomForestRegressor()
    rf.fit(X, Y)

    Feat[i] = (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))
imp_feat = {}
for i in Feat.keys():
    imp_feat[i] = set()
    for j in Feat[i]:
        if(j[0]>0.0001):
            imp_feat[i].add(j[1])
Feat_selected = imp_feat[0]

for i in imp_feat.keys():
    Feat_selected = Feat_selected|imp_feat[i]

#for xy in list(Train.columns.values):
#    print(xy)
##########################################################################
# Feature Selection Done !!

Train_RF = Train[list(Feat_selected)]
Train_RF['SalePrice'] = Sp
Test_RF = Test[list(Feat_selected)]
Train_RF['Log_SalePrice'] = np.log(Train_RF['SalePrice'])

#################################################################################################
###########################################################################


SVR_gamma = ["rbf","poly","sigmoid"]
#SVR_Kernel = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
SVR_Kernel = ["poly"]
SVR_C = np.arange(0.1,1,0.1)# : float, optional (default=0.1)
SVR_epsilon=np.arange(0.1,1,0.2)

###########################################################################
Count = 1
Error = pd.DataFrame(columns=('kernel ','c ','epsilon ','ERROR'))

for i in range(3):
    for kernel in SVR_Kernel:
        for error in SVR_C:
            for eps in SVR_epsilon: 
                msk = np.random.rand(len(Train_RF)) < 0.65
                train = Train_RF[msk]
                test = Train_RF[~msk]
                Sp_train = train['Log_SalePrice']
                Sp_test = test['Log_SalePrice']
                
                train.drop('SalePrice',1)
                test.drop('SalePrice',1)
                train.drop('Log_SalePrice',1)
                test.drop('Log_SalePrice',1)
                
#                est = SVR(gamma=gamma,C=error,epsilon = eps)
                est = SVR(kernel=kernel, C=error,epsilon = eps)
                est.fit(train, Sp_train)
                Result = est.predict(test)
                    
                Res  = pd.DataFrame(index = test.index)
                Res['Predicted']  = np.exp(Result)
                Res['Expected'] = np.exp(Sp_test)
                Res['Error'] = np.log(Res.Predicted) - np.log(Res.Expected)
                Res['Error'] = np.power(Res.Error,2)
#               print(Res.tail())
                    
                Error.loc[Count] = [kernel, error, eps,(np.round(np.sqrt(np.mean(Res.Error)),3))]
                Count += 1
            print("")
            print(Error.tail(8))
Error.to_csv("Error_SVR_RF.csv", sep=',', encoding='utf-8')
print("\n\n\n\n\nRF done")
###########################################################################