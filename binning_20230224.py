#pip install scorecardpy


# Step 1:Data Preparation
import scorecardpy as sc
df = sc.germancredit()
df['Target'] = df.apply(lambda x:1 if x.creditability == 'bad' else 0, axis = 1)
df = df.drop('creditability', axis=1)


# Step 2: Create And Review Binning
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
        woe = woe.assign(Dtype = dtype)
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

    woe_summary = woe_summary[['Variable_Name','Dtype','Group_IV','Bin','Count','Count (%)','Non-event','Event','Event rate','WoE','IV','JS']]

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
        
    return woe_summary,all_new_x,optimized_woe_summary,optimized_x_woe

#Apply Function of Binning

woe_summary,all_new_x,optimized_woe_summary,optimized_x_woe = my_woebin(df,'Target',min_iv=0.02)




# Step 3: Create step_logit
def step_logit(x,y,vif_cap=5):
    #remove VIF >= 5 parameter
    #remove lowest abs coeff parameter first
    import statsmodels.api as sm
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    x_c = sm.add_constant(x)
    
    print('-----------Start VIF Checking-----------')
    sm_lr = sm.Logit(y,x_c).fit(disp=0)
    vif = pd.DataFrame()
    model_params = sm_lr.params.index[1:]
    model_coeff = sm_lr.params[1:].reset_index().drop('index',axis = 1)
    vif["VIF Factor"] = [variance_inflation_factor(x[model_params].values, i) for i in range(x[model_params].shape[1])]
    vif["features"] = x[model_params].columns
    vif["abs_coeff"] = abs(model_coeff)
    vif = vif.sort_values(by = ['abs_coeff'],ascending=[True])
    del_col_vif = list(vif[vif['VIF Factor'] >=vif_cap].reset_index().drop('index',axis = 1).features)
    
    while len(del_col_vif) >0:
        print('VIF Remove:' + str(del_col_vif[0]))
        del x_c[del_col_vif[0]]
        sm_lr = sm.Logit(y,x_c).fit(disp=0)
        vif = pd.DataFrame()
        model_params = sm_lr.params.index[1:]
        model_coeff = sm_lr.params[1:].reset_index().drop('index',axis = 1)
        vif["VIF Factor"] = [variance_inflation_factor(x[model_params].values, i) for i in range(x[model_params].shape[1])]
        vif["features"] = x[model_params].columns
        vif["abs_coeff"] = abs(model_coeff)
        vif = vif.sort_values(by = ['abs_coeff'],ascending=[True])
        del_col_vif = list(vif[vif['VIF Factor'] >=5].reset_index().drop('index',axis = 1).features)
    

    #Do forward Stepwise regression first
    print('-----------Start Forward Regression based on BIC-----------')
    test_list =[]
    test_bic = None
    col = list(vif.sort_values(by = ['abs_coeff'],ascending=[False]).reset_index().drop('index',axis = 1).features)
    col.insert(0, 'const')

    for add_v in col:
        test_list.append(add_v)
        sm_lr = sm.Logit(y,x_c[test_list]).fit(disp=0)
        p_value_over = sm_lr.pvalues[sm_lr.pvalues>=0.05].sort_values(ascending=False)
        if len(p_value_over)==0:
            if test_bic ==None:
                test_list = test_list 
                test_bic = sm_lr.bic
                print('Add:'+str(add_v)+', BIC = ' + str(sm_lr.bic))            
                
            elif sm_lr.bic<test_bic:
                test_list = test_list 
                test_bic = sm_lr.bic    
                print('Add:'+str(add_v)+', BIC = ' + str(sm_lr.bic))
            else:
                test_list.remove(add_v)
                test_bic = test_bic
                print('Remove:'+str(add_v)+', Worse BIC = ' + str(sm_lr.bic))
        else:
                test_list.remove(add_v)
                test_bic = test_bic
                print('Remove:'+str(add_v)+' due to insignificant (p-value too large)')
            

    x_c = x_c[test_list]
    best_model = sm.Logit(y,x_c).fit(disp=0) 
    
    vif = pd.DataFrame()
    model_params = best_model.params.index[1:]
    vif["VIF Factor"] = [variance_inflation_factor(x_c[model_params].values, i) for i in range(x_c[model_params].shape[1])]
    vif["features"] = x_c[model_params].columns
    vif = vif.sort_values(by = ['VIF Factor'],ascending=[False]).reset_index().drop('index',axis = 1)
       
    print('-----------Best Model-----------')
    print(best_model.summary2())
    print(vif)
    
    return best_model

y = df['Target']
best_model = step_logit(optimized_x_woe,y)




# Step 4: Create Scorecard

def my_scorecard(best_model,optimized_woe_summary,a,b,reverse=1):
    import pandas as pd
    min_woe = pd.DataFrame(optimized_woe_summary.groupby(['Variable_Name'])['WoE'].min().rename('Min_Woe')).reset_index().rename(columns = {'index':'Variable_Name'})
    min_woe['Variable_Name'] = min_woe['Variable_Name']+'_WOE'
    
    max_woe = pd.DataFrame(optimized_woe_summary.groupby(['Variable_Name'])['WoE'].max().rename('Max_Woe')).reset_index().rename(columns = {'index':'Variable_Name'})
    max_woe['Variable_Name'] = max_woe['Variable_Name']+'_WOE'
    
    min_max_tbl = pd.DataFrame(pd.Series(best_model.params,name = 'Coeff')).reset_index().rename(columns = {'index':'Variable_Name'})
    
    min_max_tbl = pd.merge(min_max_tbl,min_woe, how='left',on = ['Variable_Name'])
    min_max_tbl = pd.merge(min_max_tbl,max_woe, how='left',on = ['Variable_Name'])

    min_max_tbl = min_max_tbl.fillna(1)
    min_max_tbl = min_max_tbl.assign(t1 = min_max_tbl.Coeff * min_max_tbl.Min_Woe)
    min_max_tbl = min_max_tbl.assign(t2 = min_max_tbl.Coeff * min_max_tbl.Max_Woe)
    min_max_tbl = min_max_tbl.assign(Min = min_max_tbl[['t1','t2']].min(axis=1))
    min_max_tbl = min_max_tbl.assign(Max = min_max_tbl[['t1','t2']].max(axis=1))

    smin = sum(min_max_tbl.Min)
    smax = sum(min_max_tbl.Max)
    intercept = min_max_tbl.Coeff.iloc[0]


    if reverse == 0:
        slope = (b-a) / (smax - smin)
        shift = a - slope * smin
    if reverse == 1:
        slope = (a-b) / (smax - smin)
        shift = b - slope * smin
    base_points = shift + slope * intercept
    #Create Scorecard

    scorecard = optimized_woe_summary.assign(Variable_Name2 = optimized_woe_summary.Variable_Name + "_WOE")
    scorecard = scorecard.loc[scorecard['Variable_Name2'].isin(list(min_max_tbl.Variable_Name))]
    min_max_tbl2 = min_max_tbl.rename(columns = {'Variable_Name': 'Variable_Name2'})
    scorecard = pd.merge(scorecard,min_max_tbl2[['Variable_Name2','Coeff']],how='left',on = 'Variable_Name2')
    scorecard = scorecard.assign(Odds = scorecard.WoE* scorecard.Coeff)
    n = len(best_model.params) - 1
    scorecard = scorecard.assign(Points = base_points/n + slope*scorecard.Odds)
    scorecard = scorecard.drop('Variable_Name2',axis = 1)
    scorecard = scorecard[(scorecard['Bin'].astype(str) != 'Special') & (scorecard['Count'] != '0')]
    return scorecard,slope,shift


scorecard,slope,shift = my_scorecard(best_model,optimized_woe_summary,0,1000,reverse=1)







def apply_scorecard(df,scorecard,shift,slope):
    all_variable = set(scorecard.Variable_Name)
    #all_variable = ['TTL_NUM_LOAN_ACTIVE']
    #df =data
    import pandas as pd
    import math
    data_map = {}
    point_list=[]
    for var in all_variable:
        print(var+' Transforming...')
        data_map = {}
        data_type = list(scorecard[scorecard.Variable_Name == var]['Dtype'])[0]
        
        if data_type == 'categorical':
            comb_temp = scorecard[scorecard.Variable_Name == var]['Bin'].tolist()
            for i in range(0,len(comb_temp)):
                item_temp = comb_temp[i]
    
                if str(item_temp) == 'Missing':
                    item = None
                    data_map_temp = {item: str(item_temp)}
                    data_map.update(data_map_temp)               
    
                else:
                    for j in range(0,len(item_temp)):
                        item = item_temp[j]
                        data_map_temp = {item: str(item_temp)}
                        data_map.update(data_map_temp)
            
            df = df.assign(temp = df[var].map(data_map))\
                .rename(columns={'temp':str(var)+'_Bin'})

            df[str(var)+'_Bin']=df[var].map(data_map).astype('string')
            df[str(var)+'_Bin'] = df[str(var)+'_Bin'].fillna('Missing', inplace=False)
    
    
        if data_type == 'numerical':
            #get cut off point (upper bound) in comb_temp
            comb_temp = pd.DataFrame(scorecard[scorecard.Variable_Name == var]['Bin'])
            comb_temp[['LARGER_OR_EQUAL', 'LESS_THEN']] = comb_temp['Bin'].apply(lambda x: pd.Series(str(x).split(", ")))
            comb_temp = comb_temp.drop('LARGER_OR_EQUAL', axis = 1)
            comb_temp = comb_temp.dropna()
            comb_temp['LESS_THEN'] = comb_temp['LESS_THEN'].str.rstrip(')').astype(float)
            #comb_temp['LESS_THEN'] = comb_temp['LESS_THEN'].apply(lambda x: "{:.2f}".format(x))
            
            #convert comb_temp to a dictionary
            data_map_temp = list(comb_temp['LESS_THEN'])
            
            #transform comb_temp to a correct bin
            
            comb_temp2=pd.cut(comb_temp['LESS_THEN']-1, right = False, bins = [float('-inf')] + data_map_temp).astype('string')
            comb_temp3=pd.cut(comb_temp['LESS_THEN'], right = False, bins = [float('-inf')] + data_map_temp).astype('string')
            adj_comb_temp = pd.concat([comb_temp2,comb_temp3],axis = 0).reset_index().drop('index', axis = 1)
            adj_comb_temp = adj_comb_temp.drop_duplicates()
            adj_comb_temp = adj_comb_temp[adj_comb_temp.LESS_THEN != 'None']
            
            comb_temp = comb_temp.reset_index().drop(['index','LESS_THEN'], axis = 1)
            adj_comb_temp = adj_comb_temp.reset_index().drop('index', axis = 1)
            comb_temp = pd.concat([comb_temp,adj_comb_temp],axis = 1)
            comb_temp['LESS_THEN'].fillna('[-inf, inf)',inplace = True)
            comb_temp = comb_temp.rename(columns = {'LESS_THEN':str(var)+'_Bin_TEMP'})
            comb_temp = comb_temp.rename(columns = {'Bin':str(var)+'_Bin'})

            
            df=df.assign(temp = pd.cut(df[var], right = False, bins = [float('-inf')] + data_map_temp))\
                .rename(columns={'temp':str(var)+'_Bin_TEMP'})
            df[str(var)+'_Bin_TEMP'] = df[str(var)+'_Bin_TEMP'].astype('string')
            
                              
            #replace adj bin after cut
            df = pd.merge(df,comb_temp, how = 'left', on = [str(var)+'_Bin_TEMP']).drop(str(var)+'_Bin_TEMP',axis = 1)
            
            df[str(var)+'_Bin'] = df[str(var)+'_Bin'].fillna('Missing', inplace=False)  
    
        
        score_pt = scorecard[scorecard.Variable_Name == var][['Bin','Points']]
        score_pt.Bin = score_pt.Bin.astype('string')
        score_pt.Points = score_pt.Points.astype('float')
        score_pt = score_pt.rename(columns={'Bin': str(var)+'_Bin'})
        score_pt = score_pt.rename(columns={'Points': str(var)+'_Points'})
        df = pd.merge(df,score_pt,how = 'left',on = str(var)+'_Bin')
    
        point_list.append(str(var)+'_Points')
        print(var+' Done')
    
    df['Total_Points'] = df[point_list].sum(axis = 1)
    df['Odds'] = (df['Total_Points']-shift)/slope
    df['Prob'] = 1/(1+math.e**(df['Odds']*-1))
    
    new = df[['Total_Points','Odds','Prob']]
    return new

#test_new = apply_scorecard(test,scorecard,shift,slope)

df_new = apply_scorecard(df,scorecard,shift,slope)

df_new.Total_Points







# Step 5: Plot ROC Curve

def plot_roc(y_label,y_pred):
    from sklearn import metrics 
    import matplotlib.pyplot as plt 
    """
    蝏roc?脩瑪
    param:
        y_label -- ?????list/array
        y_pred -- 憸????list/array
    return:
        roc?脩瑪
    """
    tpr,fpr,threshold = metrics.roc_curve(y_label,y_pred) 
    AUC = metrics.roc_auc_score(y_label,y_pred) 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tpr,fpr,color='blue',label='AUC=%.3f'%AUC) 
    ax.plot([0,1],[0,1],'r--')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show(ax)

y = df['Target']
y_pred = df_new.Prob
plot_roc(y,y_pred)










# Step 6: Plot KS Diagram

def plot_model_ks(y_label,y_pred):
    import matplotlib.pyplot as plt 
    """
    蝏ks?脩瑪
    param:
        y_label -- ?????list/array
        y_pred -- 憸????list/array
    return:
        ks?脩瑪
    """
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)

plot_model_ks(y,y_pred)








import pandas as pd

df_new2 = pd.concat([df_new,y],axis = 1)
df_new2

non_event = df_new2[df_new2.Target == 0]['Total_Points']
event = df_new2[df_new2.Target == 1]['Total_Points']

def plot_hist_score(non_event,event):
    import matplotlib.pyplot as plt
    plt.hist(non_event, label="non-event", color="b", alpha=0.35)
    plt.hist(event, label="event", color="r", alpha=0.35)
    plt.xlabel("score")
    plt.legend()
    plt.show()
    

plot_hist_score(non_event,event)    
