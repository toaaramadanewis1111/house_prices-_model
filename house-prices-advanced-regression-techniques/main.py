import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer #smart value filling tool
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
print(train.shape)
print(test.shape)
print(train.head())
print(test.head())
#hn3ml save ll id 3shan n3ml submission b3d kda hnms7ha mn el data
test_ids = test["Id"].copy()
train = train.drop(columns=["Id"])
test  = test.drop(columns=["Id"])

#features and target split alshan n3ml train ll model
x=train.drop('SalePrice',axis=1)
Y=np.log1p(train['SalePrice'])

#missingvalues
missing=x.isnull().sum()
print(missing[missing>0])

#filling missing values
#number mean and catg mode 

num=x.select_dtypes(include=[np.number]).columns
cat=x.select_dtypes(include=[object]).columns
num_imp=SimpleImputer(strategy='median')#median is better for numerical data because it is less affected by outlier
cat_imp=SimpleImputer(strategy='most_frequent')#ll categorical htb2a most freq
x[num]=num_imp.fit_transform(x[num])
x[cat]=cat_imp.fit_transform(x[cat])
test[num]=num_imp.transform(test[num])
test[cat]=cat_imp.transform(test[cat])

#feature eng
# estkhdmna total house size , house agre etxc.. alshana el model ylearn kwyss mn el data dy
def add_features(df):
    df = df.copy()
    df["TotalSF"]= df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"]=(df["FullBath"] + 0.5 * df["HalfBath"] +df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorch"]=(df["OpenPorchSF"] + df["EnclosedPorch"] +df["3SsnPorch"]+ df["ScreenPorch"])
    df["HouseAge"]=df["YrSold"] - df["YearBuilt"]
    df["RemodAge"]=df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"]=df["YrSold"] - df["GarageYrBlt"].fillna(df["YearBuilt"])
    df["IsRemodeled"]=(df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["QualxSF"]=df["OverallQual"] * df["TotalSF"]
    df["OverallScore"]=df["OverallQual"] * df["OverallCond"]
    df["AreaQuality"]=df["GrLivArea"] * df["OverallQual"]
    df["AgeQuality"]=df["HouseAge"] * df["OverallQual"]
    df["TotalQuality"]=df["TotalSF"] * df["OverallQual"]
    return df

x    = add_features(x)
test = add_features(test)

#encoding
for col in x.select_dtypes(include=["object"]).columns:
    x[col]=x[col].astype("category")
    test[col]=test[col].astype("category")

X=x #KAN FY ERROr fa amlt intialize el X b x


#kfold  4 training w wahda validation
kf=KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds=np.zeros(len(test))
xgb_preds=np.zeros(len(test))
rmse_list=[]
#tuning 
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val=X.iloc[train_idx], X.iloc[val_idx] # split data to train and validation
    y_train, y_val=Y.iloc[train_idx], Y.iloc[val_idx]
#lightgbm model byt3lm size quailty , locations
    lgb_model=LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
#nfs lgm bs fy tnwo3 
    xgb_model=XGBRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    enable_categorical=True
)
    xgb_model.fit(X_train, y_train)
    lgb_val=lgb_model.predict(X_val)
    rmse=np.sqrt(mean_squared_error(y_val, lgb_val))
    rmse_list.append(rmse)
    print("fold",fold+1,"rmse:",rmse)
    lgb_preds+=lgb_model.predict(test) / 5#avg ll 5 folds
    xgb_preds+=xgb_model.predict(test) / 5
print("rmse:",np.mean(rmse_list))
#full data
final_preds=(lgb_preds + xgb_preds) / 2
predictions=np.expm1(final_preds) #exponential function alshan nrg3hlaa mn el elog
sub=pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})
sub.to_csv("sub.csv", index=False)
#GUI
import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("House Price Predictor")

overall_qual = st.slider("Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area")

if st.button("Predict"):
    X = np.array([[overall_qual, gr_liv_area]])
    pred = model.predict(X)
    st.write(np.expm1(pred)[0])
