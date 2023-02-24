import scorecardpy as sc
df = sc.germancredit()
df['Target'] = df.apply(lambda x:1 if x.creditability == 'bad' else 0, axis = 1)
df = df.drop('creditability', axis=1)


def my_woebin(df,y,break_list={},min_iv=0.02):
    target = y
    import pandas as pd
    import numpy as np
    from optbinning import OptimalBinning
    ttl_var = df.columns.tolist()
    ttl_var.remove(target)
    woe_summary = []
    all_new_x = []
    for variable in ttl_var:
        variable = variable
        target = "Target"
        x = df[variable].values
        y = df[target].values

        if x.dtype == 'float64' or x.dtype =='int64':
            dtype = "numerical"
        else:
            dtype = "categorical"

        #break_list={'ACCOM_CODE':[]}

        if variable in break_list:
            if break_list[variable] == []:
                dtype = "categorical"
                user_splits = sorted(np.array(pd.DataFrame(set({item for item in x if item is not None}))))
                user_splits_fixed = list([True] * len(user_splits))
            else:
                if dtype == "categorical":
                    user_splits = np.array(break_list[variable], dtype=object)
                else:
                    user_splits = break_list[variable]
                user_splits_fixed = list([True] * len(user_splits))
        else:
            user_splits = None
            user_splits_fixed = None

        optb = OptimalBinning(name=variable, dtype=dtype, solver="cp",user_splits=user_splits,user_splits_fixed=user_splits_fixed,split_digits=0,monotonic_trend="ascending")
        optb.fit(x, y)
        binning_table = optb.binning_table

       
        woe = binning_table.build()
        IV = woe[-1:].IV[0]
        woe = woe.assign(Group_IV = IV)
        woe = woe.assign(Variable_Name = variable)
        woe.drop(woe.index[-1],axis = 0,inplace=True)

       

        x_transform_bins = optb.transform(x, metric="bins")
        x_transform_woe = optb.transform(x, metric="woe")
        new_x = pd.DataFrame({variable: x,variable+'_BINS': x_transform_bins, variable+'_WOE': x_transform_woe})

        if len(woe_summary)==0:
            woe_summary = woe
            all_new_x = new_x
        else:
            woe_summary = pd.concat([woe_summary,woe],axis = 0)
            all_new_x = pd.concat([all_new_x,new_x],axis = 1)

        binning_table.plot(metric="event_rate", show_bin_labels=True)

        print((variable + " binned with IV "+str(round(IV,4))))

    
    woe_summary = woe_summary[['Variable_Name','Group_IV','Bin','Count','Count (%)','Non-event','Event','Event rate','WoE','IV','JS']]

    col_name_list = []
    for i in set(woe_summary.loc[woe_summary.Group_IV >= min_iv].Variable_Name):
        col_name = i + "_WOE"
        col_name_list.append(col_name)

    if len(col_name)>0:
        optimized_woe_summary = woe_summary.loc[woe_summary.Group_IV >= min_iv]
        optimized_x_woe = all_new_x[col_name_list]
 
        print('Here are '+ str(len(col_name_list)) +' features with optimized IV'+str(list(col_name_list)))
    else:
        optimized_x_woe = []
        optimized_woe_summary = []
        print('No optimized IV, please adjust min_iv')
        
    return woe_summary,all_new_x,optimized_woe_summary,optimized_x_woe,optb

 
woe_summary,new_x,optimized_woe_summary,optimized_x_woe = my_woebin(df,y = "Target")



y = df['Target']

def step_logit(x,y):
    import statsmodels.api as sm
    x_c = sm.add_constant(x)
    
    sm_lr = sm.Logit(y,x_c).fit()
    del_col = sm_lr.pvalues[sm_lr.pvalues>=0.05].sort_values(ascending=False)
    
     
    while len(del_col) >0:
        del x_c[del_col.index[0]]
        sm_lr = sm.Logit(y,x_c).fit()
        del_col = sm_lr.pvalues[sm_lr.pvalues>=0.05].sort_values(ascending=False)
    
    best_model = sm_lr
    best_model.bic
    print(best_model.summary2())
    return best_model

best_model = step_logit(optimized_x_woe,y)

min_woe = pd.Series(optimized_x_woe.min(axis = 0),name ='Min_Woe')
max_woe = pd.Series(optimized_x_woe.max(axis = 0),name = 'Max_Woe')
min_max_tbl = pd.Series(best_model.params,name = 'Coeff')

min_max_tbl = pd.merge(min_max_tbl,min_woe, how='left', left_index=True, right_index=True)
min_max_tbl = pd.merge(min_max_tbl,max_woe, how='left', left_index=True, right_index=True)
min_max_tbl = min_max_tbl.fillna(1)
min_max_tbl = min_max_tbl.assign(t1 = min_max_tbl.Coeff * min_max_tbl.Min_Woe)
min_max_tbl = min_max_tbl.assign(t2 = min_max_tbl.Coeff * min_max_tbl.Max_Woe)
min_max_tbl = min_max_tbl.assign(Min = min_max_tbl[['t1','t2']].min(axis=1))
min_max_tbl = min_max_tbl.assign(Max = min_max_tbl[['t1','t2']].max(axis=1))

smin = sum(min_max_tbl.Min)
smax = sum(min_max_tbl.Max)
intercept = min_max_tbl.Coeff.iloc[0]

a = 0
b = 1000
slope = 1 * (a - b) / (smax - smin)
shift = b - slope * smin


base_points = shift + slope * intercept







new_points = base_points / n + slope * points



A = score-B*np.log(odds)

score = shift+slope*smax















#define shift and slope
import numpy as np
min_odd = sum(min_max_tbl.Coeff * min_max_tbl.Min)
pdo = 20
slope = pdo / np.log(2)
shift = 400 - slope * np.log(min_odd)

max_odd = sum(min_max_tbl.Coeff * min_max_tbl.Max)
shift + slope*np.log(min_odd)


shift + slope*np.log(max_odd)





def _compute_scorecard_points(points, binning_tables, method, method_data,
                              intercept, reverse_scorecard):
    """Apply scaling method to scorecard."""
    n = len(binning_tables)

    sense = -1 if reverse_scorecard else 1

    if method == "pdo_odds":
        pdo = method_data["pdo"]
        odds = method_data["odds"]
        scorecard_points = method_data["scorecard_points"]

        factor = pdo / np.log(2)
        offset = scorecard_points - factor * np.log(odds)

        new_points = -(sense * points + intercept / n) * factor + offset / n
    elif method == "min_max":
        a = method_data["min"]
        b = method_data["max"]

        min_p = np.sum([np.min(bt.Points) for bt in binning_tables])
        max_p = np.sum([np.max(bt.Points) for bt in binning_tables])

        smin = intercept + min_p
        smax = intercept + max_p

        slope = sense * (a - b) / (smax - smin)
        if reverse_scorecard:
            shift = a - slope * smin
        else:
            shift = b - slope * smin

        base_points = shift + slope * intercept
        new_points = base_points / n + slope * points

    return new_points







 

#https://www.youtube.com/watch?v=M_iaBcLEN-8

Shift and slope - to form constant range of scroe

 

 

Base point = shift + slope * intercept

 

 

point of j predictor = (base point / num of predictor ) + slope * odd_ratio

 

given pdo = 20 (double odd at every 20 points)

 

points = shift +slope * ln(odd)

points + 20 = shift + slope *ln(2 * odd)

 

 

So, we can get

 

slope = pdo / in(2)

shift = point - slope * ln(odds)

 

def cal_scale(score,odds,PDO,model):

    import numpy as np

    """

    计算分数校准的A，B值，基础分

    param:

        odds：设定的坏好比 float

        score: 在这个odds下的分数 int

        PDO: 好坏翻倍比 int

        model:模型

    return:

        A,B,base_score(基础分)

    """

    B = 20/(np.log(odds)-np.log(2*odds))

    A = score-B*np.log(odds)

    base_score = A+B*model.intercept_[0]

    return A,B,base_score

 

 

score=400,odds=999/1,pdo=20


import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import HuberRegressor

from optbinning import BinningProcess
from optbinning import Scorecard
scorecard = Scorecard(binning_process=binning_process,
                      estimator=estimator, scaling_method="min_max",
                      scaling_method_params={"min": 0, "max": 100},
                      reverse_scorecard=True)










def _compute_scorecard_points(points, binning_tables, method, method_data,
                              intercept, reverse_scorecard):
    """Apply scaling method to scorecard."""
    n = len(binning_tables)

    sense = -1 if reverse_scorecard else 1

    if method == "pdo_odds":
        pdo = method_data["pdo"]
        odds = method_data["odds"]
        scorecard_points = method_data["scorecard_points"]

        factor = pdo / np.log(2)
        offset = scorecard_points - factor * np.log(odds)

        new_points = -(sense * points + intercept / n) * factor + offset / n
    elif method == "min_max":
        a = method_data["min"]
        b = method_data["max"]

        min_p = np.sum([np.min(bt.Points) for bt in binning_tables])
        max_p = np.sum([np.max(bt.Points) for bt in binning_tables])

        smin = intercept + min_p
        smax = intercept + max_p

        slope = sense * (a - b) / (smax - smin)
        if reverse_scorecard:
            shift = a - slope * smin
        else:
            shift = b - slope * smin

        base_points = shift + slope * intercept
        new_points = base_points / n + slope * points

    return new_points