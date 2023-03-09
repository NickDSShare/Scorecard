import scorecardpy as sc
df = sc.germancredit()
df['Target'] = df.apply(lambda x:1 if x.creditability == 'bad' else 0, axis = 1)
df = df.drop('creditability', axis=1)
df.head(5)


# Step 1: Create my_woebin
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

woe_summary,new_x,optimized_woe_summary,optimized_x_woe = my_woebin(df,y = "Target")


#WOE can be generated
optimized_x_woe.head(10)

 
# Step 2: Create step_logit
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

y = df['Target']
best_model = step_logit(optimized_x_woe,y)

# Create my_scorecard
#Requrie all elements from previous steps: optimized_x_woe,optimized_woe_summary,best_model

def my_scorecard(optimized_x_woe,best_model,optimized_woe_summary,a,b):
    import pandas as pd
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

    a = a
    b = b
    slope = 1 * (a - b) / (smax - smin)
    shift = b - slope * smin
    base_points = shift + slope * intercept
    #Create Scorecard

    scorecard = optimized_woe_summary.assign(Variable_Name2 = optimized_woe_summary.Variable_Name + "_WOE")
    scorecard = scorecard.loc[scorecard['Variable_Name2'].isin(list(min_max_tbl.index))]
    min_max_tbl2 = min_max_tbl.reset_index()
    min_max_tbl2 = min_max_tbl2.rename(columns = {'index': 'Variable_Name2'})
    scorecard = pd.merge(scorecard,min_max_tbl2[['Variable_Name2','Coeff']],how='left',on = 'Variable_Name2')
    scorecard = scorecard.assign(Odds = scorecard.WoE* scorecard.Coeff)
    n = len(best_model.params) - 1
    scorecard = scorecard.assign(Points = base_points/n + slope*scorecard.Odds)
    scorecard = scorecard.drop('Variable_Name2',axis = 1)
    scorecard = scorecard[(scorecard['Bin'].astype(str) != 'Special') & (scorecard['Count'] != '0')]
    return scorecard

scorecard = my_scorecard(optimized_x_woe,best_model,optimized_woe_summary,0,100)

scorecard.head(50)



#If I have new data, want to predict Y by whole new data
test = df

#list all required columns from Scorecard
all_variable = set(scorecard.Variable_Name)

 
import pandas as pd
data_map = {}
for var in all_variable:
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

        test[str(var)+'_Bin']=test[var].map(data_map).astype(str)
        score_pt = scorecard[scorecard.Variable_Name == var][['Bin','Points']]
        score_pt.Bin = score_pt.Bin.astype('string')
        score_pt.Points = score_pt.Points.astype('float')
        score_pt = score_pt.rename(columns={'Bin': str(var)+'_Bin'})
        score_pt = score_pt.rename(columns={'Points': str(var)+'_Points'})


        test[str(var)+'_Bin'] = test[str(var)+'_Bin'].astype('string')
        test = pd.merge(test,score_pt,how = 'left',on = str(var)+'_Bin')
        
      
        
    
# Python program to demonstrate creating
# pandas Dataframe from lists using zip.
 
import pandas as pd
 
# List1
Name = ['tom', 'krish', 'nick', 'juli']
 
# List2
Age = [25, 30, 26, 22]
 
# get the list of tuples from two lists.
# and merge them by using zip().
list_of_tuples = list(zip(Name, Age))
 
# Assign data to tuples.
list_of_tuples
        

str(item_temp)

 

comb_temp = scorecard[scorecard.Variable_Name == var]['Bin'].tolist()

 

 

 

 

len(c[0])

 

c[2]

 

len(c)

 

np.array_split(c,1)

 

 

#Handling numerical first

numerical_card = scorecard[scorecard.Dtype == 'numerical'].reset_index()

numerical_card = numerical_card.drop('index', axis = 1)

 

numerical_card[['LARGER_OR_EQUAL', 'LESS_THEN']] = numerical_card["Bin"].apply(lambda x: pd.Series(str(x).split(", ")))

numerical_card = numerical_card.drop('LARGER_OR_EQUAL', axis = 1)

numerical_card['LESS_THEN'] = numerical_card['LESS_THEN'].str.rstrip(')').astype(float)

 

list(numerical_card['LESS_THEN'])

num_map = {}

for i in set(numerical_card.Variable_Name):

    the_list = list(numerical_card[numerical_card.Variable_Name == i]['LESS_THEN'])

    the_list = [x for x in the_list if np.isnan(x) == False]

    num_map_temp = {i:the_list}

    num_map.update(num_map_temp)






 
# Function for Turning New X to Point

 
import numpy as np
import pandas as pd
my_list = pd.DataFrame()

col_name = 'savings.account.and.bonds'
a = np.array(scorecard[scorecard.Variable_Name == col_name].Bin)
for i in a:
    if str(i) == 'Missing' or str(i) == 'Special':
        temp = pd.DataFrame(data = {'Variable_Name': [col_name],'Bin': [i], 'Bin2': [None]})
        my_list = pd.concat([my_list,temp])
    else:
        for j in i:
            temp = pd.DataFrame(data = {'Variable_Name': [col_name],'Bin': [i], 'Bin2': [j]})
            my_list = pd.concat([my_list,temp])





import pandas as pd
#Split the scorecard into two - categorical and numerical
#Handling categorical first
categorical_card = scorecard[scorecard.Dtype == 'categorical'].reset_index()
categorical_card = categorical_card.drop('index', axis = 1)
n = len(categorical_card)
if n >0:
    cat_map = pd.DataFrame()
    for i in range(0,n):
        r = pd.DataFrame(categorical_card.iloc[[i]])
        retain_Variable_Name = list(r.Variable_Name)[0]
        retain_Bin = list(r.Bin)[0]
        retain_Odds = list(r.Odds)[0]
        retain_Points = list(r.Points)[0]
        split_raw = np.array(retain_Bin)
        if str(split_raw) == 'Missing':
            temp = pd.DataFrame(data = {'Variable_Name': [retain_Variable_Name],'Bin': [retain_Bin], 'Bin2': [None], 'Odds': [retain_Odds], 'Points': [retain_Points]})
            cat_map = pd.concat([cat_map,temp])
        else:
            for j in split_raw:
                temp = pd.DataFrame(data = {'Variable_Name': [retain_Variable_Name],'Bin': [retain_Bin], 'Bin2': [j], 'Odds': [retain_Odds], 'Points': [retain_Points]})
                cat_map = pd.concat([cat_map,temp])                
            

#Handling numerical first
numerical_card = scorecard[scorecard.Dtype == 'numerical'].reset_index()
numerical_card = numerical_card.drop('index', axis = 1)

num_items = set(numerical_card.Variable_Name)
n = len(num_items)
if n >0:
    num_map = {}
    for i in num_items:
        obs = numerical_card[numerical_card.Variable_Name== i]
        retain_Variable_Name = i

        
        a = np.array(obs.Bin)
        for j in a:
            if j != 'Missing':
                temp = {i:}
                print(j.split(", ")[1][:-1])








n = len(numerical_card)

a = np.array(numerical_card.Bin)



a = np.array(numerical_card[numerical_card.Variable_Name=='duration.in.month'].Bin)
for i in a:
    if i != 'Missing':
        print(i.split(", ")[1][:-1])

categorical_card[[0]]




cat_comb = pd.DataFrame()
for v in categorical_card['Variable_Name']:
    print(v)


categorical_card['Variable_Name'].loc[categorical_card.index[0]]



 
def my_predict(df,model):
    col_list = list(best_model.params.index)
    if 'const' in col_list: df['const'] = 1
    x = df[col_list]
    pred_y = model.predict(x)
    return pred_y

   

    


tryt = new_x.sample(frac=1)

my_predict(tryt,model = best_model)













#############try

given_sequence = [2, 5, 8, 10, 15]# Given sequence

import pandas as pd
import random
# create an Empty DataFrame object
see = pd.DataFrame()
see['col1'] = [random.randint(-1000, 100) for i in range(0, 100)]


see['mapped'] = pd.cut(see['col1'], right=False, bins=[float('-inf')] + given_sequence+[float('inf')], 
                      labels=['<2', '2-5', '5-8','8-10','10-15','>15'])
