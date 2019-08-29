
# coding: utf-8

# In[4]:


'''
Function List:
    update_progress(job_title, progress):
        show status bar of long process. Operation purpose
        
    SummarizeDefault(raw_data,default_flag_column,by_vars,plot=False,continuous_var=False,n_bin=10):
        Summarize default data by different variables
        
        raw_data: pandas.DataFrame. The dataset contains the default column and groupby variable
        default_flag_column: String. column name of the default flag in dataset
        by_vars: String. Column name of the groupby variable
        continuous_var: Boolean. Default as False. If select as True, function will bucket the variable to do groupby.
        n_bin: int. If contunuous_var is set as True, function will bucket the variable into "n_bin" bins
        plot: Boolean. Default as False. If select True, function will generate graph
        
        return a DataFrame
    
    CalcROCPrep(s, default_flag_data):
        Preparation for ROC calculation. Not directly into result
        
    CalcROCStats(s,default_flag_data):
        Calculation of ROC stats
        s: pandas.DataFrame. Score generated from regression model
        default_flag_data: pandas.Dataframe. Default flag data from real transactions.
        
        return a dictionary of 3 stats
        
    CalcPredAccuracy(s,default_flag_data,n_bin=10,plot=False,is_default_rate=True):
        Compare the generated default probability with real default flag
        s: pandas.DataFrame. Score generated from regression model
        default_flag_data: pandas.Dataframe. Default flag data from real transactions.
        n_bin: int. default flag will be grouped into "n_bin" bins
        plot: Boolean. Default as False. If select True, function will generate graph
        is_default_rate. Boolean. Default as True. Select False only if this function is used to compare two series that are NOT between (0,1)
        
        return dataframe of bucket score VS actual default
        
                    CalcModelROC(raw_data,response_var,explanatory_var_1,explanatory_var_2="",explanatory_var_3="",explanatory_var_4="",explanatory_var_5="",explanatory_var_6="",explanatory_var_7="",explanatory_var_8="",explanatory_var_9="",explanatory_var_10=""):
        Directly generate ROC stats from a given model. Not suggested to use because of explanatory variable number limitation.
        raw_data: pandas.DataFrame. The dataset contains all variables used in model generation.
        response_var: string. Column name of response var.
        explanatory_var_1:string. Column name of response var.
        explanatory_var_2-10:string.Optional. Column name of response var.
        
        return dataframe
    
    CalcWOE_IV(data,default_flag_column,categorical_column,event_list=[1,"Yes"],continuous_variable=False,n_bucket=10):
        Calculate IV of each variable
        Calculation refer to: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        data: pandas.DataFrame. dataset that contains all variables
        default_flag_column: String. column name of the default flag in dataset
        categorical_column:String. column name of the categorical column
        continuous_var: Boolean. Default as False. If select as True, function will bucket the variable to do groupby.
        n_bin: int. If contunuous_var is set as True, function will bucket the variable into "n_bin" bins
        event_list: list. The list contains all possibility than can be considered as "event"
        
        return [IV value of certain variable, table of calculation]
    
    PerformRandomCV(data,response_var,explanatory_var_list=[],k=20,plot=True,sampling_as_default_percentage=False):
        Do Random Cross Validation of model to check model stability. 
        Return standard deviation of each coefficients.
        raw_data: pandas.DataFrame. The dataset contains all variables used in model generation.
        response_var: string. Column name of response var.
        explanatory_var_list: list. List of all names of explanatory variables
        k: int. 1/k samples will be eliminated as train set.
        plot: Boolean. Default as True. Plot std. dev of each coefficient.
        sampling_as_default_percentage: Boolean. Default as False. If selected as True, sampling will be done as the default rate.
        
        return std.dev of coefficients during k-time regression
        
    PerformK_Fold(data,response_var, explanatory_var_list=[],k=10):
        Perform k-fold cross validation
        data: pandas.DataFrame. dataset that contains all variables.
        response_var: string. column name of response variable
        explanatory_var_list: list of strings. list that contains all explanatory variables
        
        return array of each fold score
        
    BestSubsetSelection(data,response_var,explanatory_var_list=[],plot=True):
        Run best subset selection
        data: pandas.DataFrame. raw dataset
        response_var: string. Column name of response var
        explanatory_var_list: list of string. Names of columns of all explanatory candidates.
        plot: If selected as True, it will plot the selection steps graph.
        
        return all combinations' RSS and R_squared
        
    ForwardStepwiseSelection(data,response_var,explanatory_var_list=[],plot=True):
        Do forward stepwise selection
        data: pandas.DataFrame. raw dataset
        response_var: string. Column name of response var
        explanatory_var_list: list of string. Names of columns of all explanatory candidates.
        plot: if selected as True, it will plot the selection steps graph
        
        return dataframe with AIC,BIC,R_squared of each variable combination
        
    CalcSpearmanCorrelation(data,var_1,var_2):
        Calculate spearman Correlation
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        
        return float

    CalcKendallTau(data,var_1,var_2):
        Calculate KendallTau
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        
        return float
        
    CalcSamplingGoodmanKruskalGamma(data,var_1, var_2,n_sample=1000):
        Calculate GoodmanKruskalGamma estimator with sampling data. Not suggest to use into report as this is just an estimator.
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        
        return float
    
    CalcSomersD(data,var_1,var_2):
        Calculate Somers' D
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        
        return float
        
    GenerateRankCorrelation(data,var_1,var_2):
        Calculate all 4 ranked correlation.
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2. The order of var_1 and var_2 can not change.
        
        return float
'''
class ValidationTool:
    import pandas as pd
    import numpy as np
    def __init__(self):
        pass
    
    def update_progress(job_title, progress):
        '''
        show status bar of long process
        '''
        import time, sys
        length = 20 
        block = int(round(length*progress))
        msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 1))
        if progress >= 1: msg += " DONE\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()
    
    def SummarizeDefault(raw_data,default_flag_column,by_vars,plot=False,continuous_var=False,n_bin=10):
        '''
        Summarize default data by different variables
        raw_data: pandas.DataFrame. The dataset contains the default column and groupby variable
        default_flag_column: String. column name of the default flag in dataset
        by_vars: String. Column name of the groupby variable
        continuous_var: Boolean. Default as False. If select as True, function will bucket the variable to do groupby.
        n_bin: int. If contunuous_var is set as True, function will bucket the variable into "n_bin" bins
        plot: Boolean. Default as False. If select True, function will generate graph
        '''
        import numpy as np
        import pandas as pd
        data=raw_data[[default_flag_column,by_vars]]
        max_by_vars=max(data[by_vars])
        min_by_vars=min(data[by_vars])
        if continuous_var==False:
            DefaultNum=pd.DataFrame(data[[default_flag_column,by_vars]].groupby(by_vars).agg("sum"))
            DefaultNum.columns=["DefaultNum"]
            GroupTotalNum=pd.DataFrame(data.groupby(by_vars).size())
            GroupTotalNum.columns=["GroupTotalNum"]
        else:
            if raw_data[by_vars].isnull().values.any()==True:
                '''
                Follow the steps to deal with nan range:
                '''
                group_by_range=(pd.cut(np.array(data[by_vars]), np.arange(min_by_vars,max_by_vars+1,
                                                                                          (max_by_vars-min_by_vars)/n_bin),include_lowest=True)
                                                         .add_categories("missing"))
                group_by_range=group_by_range.fillna("missing")
            else:
                group_by_range=(pd.cut(np.array(data[by_vars]), np.arange(min_by_vars,max_by_vars+1,(max_by_vars-min_by_vars)/n_bin),include_lowest=True))

            DefaultNum=pd.DataFrame(data.groupby(group_by_range)[default_flag_column].sum())
            DefaultNum.columns=["DefaultNum"]
            GroupTotalNum=pd.DataFrame(data.groupby(group_by_range).size())
            GroupTotalNum.columns=["GroupTotalNum"]
        SummaryTable = pd.concat([DefaultNum, GroupTotalNum], axis=1, join_axes=[DefaultNum.index])
        SummaryTable["DefaultProb"]=(SummaryTable.DefaultNum/SummaryTable.GroupTotalNum)
        SummaryTable["Percent_of_Orbs"]=(SummaryTable.GroupTotalNum/data.shape[0])
        if plot==True:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=2, ncols=1)
            SummaryTable[["DefaultNum","GroupTotalNum"]].plot(kind="bar",grid=True,ax=axes[1],title="Number in each Group")
            SummaryTable[["DefaultProb","Percent_of_Orbs"]].plot(kind="line",grid=True,ax=axes[0],title="Percentage in each Group")
        SummaryTable["DefaultProb"]=(SummaryTable.DefaultNum/SummaryTable.GroupTotalNum).apply('{:.2%}'.format)
        SummaryTable["Percent_of_Orbs"]=(SummaryTable.GroupTotalNum/data.shape[0]).apply('{:.2%}'.format)
        return SummaryTable

    
    def CalcROCPrep(s, default_flag_data):
        '''
        Preparation for ROC calculation
        '''
        import pandas as pd
        import numpy as np
        df=pd.DataFrame({"score":100*(1-s),"outcome":default_flag_data})
        df=df.sort_values("score")
        df["cum_bad"]=df.outcome.cumsum()
        df["cum_good"]=(1-df.outcome).cumsum()
        df["cum_bad_perc"]=df.cum_bad/sum(df.outcome)
        df["cum_good_perc"]=df.cum_good/sum(1-df.outcome)
        SummaryTable=pd.DataFrame()
        SummaryTable["cum_bad"]=df[["score","cum_bad"]].groupby("score")["cum_bad"].max()
        SummaryTable["cum_good"]=df[["score","cum_good"]].groupby("score")["cum_good"].max()
        SummaryTable["cum_bad_perc"]=df[["score","cum_bad_perc"]].groupby("score")["cum_bad_perc"].max()
        SummaryTable["cum_good_perc"]=df[["score","cum_good_perc"]].groupby("score")["cum_good_perc"].max()
        return SummaryTable
    
    def CalcROCStats(s,default_flag_data):
        '''
        Calculation of ROC stats
        s: pandas.DataFrame. Score generated from regression model
        default_flag_data: pandas.Dataframe. Default flag data from real transactions.
        '''
        import numpy as np
        import pandas as pd
        df=ValidationTool.CalcROCPrep(s,default_flag_data)
        pd_rate=sum(default_flag_data)/len(default_flag_data)
        c_stat=0.5*np.dot(np.array(np.diff(([0]+(list(df.cum_good_perc))))),np.array(list(df.cum_bad_perc+df.cum_bad_perc.shift(1).fillna(0))).T)
        ar_stat=2*c_stat-1
        ks_stat=max(df.cum_bad_perc-df.cum_good_perc)

        return {"c_stat":c_stat,"ar_stat":ar_stat,"ks_stat":ks_stat}
    
    def CalcPredAccuracy(s,default_flag_data,n_bin=10,plot=False,is_default_rate=True):
        '''
        Compare the generated default probability with real default flag
        s: pandas.DataFrame. Score generated from regression model
        default_flag_data: pandas.Dataframe. Default flag data from real transactions.
        n_bin: int. default flag will be grouped into "n_bin" bins
        plot: Boolean. Default as False. If select True, function will generate graph
        is_default_rate. Boolean. Default as True. Select False only if this function is used to compare two series that are NOT between (0,1)
        '''
        import numpy as np
        import pandas as pd
        df=pd.DataFrame({"outcome":default_flag_data,"score":s})
        SummaryTable=pd.DataFrame()
        if is_default_rate==False:
            SummaryTable["actual_pd"]=(df.groupby(pd.cut(df["score"], np.arange(min(df.score),max(df.score)+1,(max(df.score)-min(df.score))/n_bin),include_lowest=True)).sum()["outcome"]/  
                          df.groupby(pd.cut(df["score"], np.arange(min(df.score),max(df.score)+1,(max(df.score)-min(df.score))/n_bin),include_lowest=True)).size())
            SummaryTable["pred_pd"]=df.groupby(pd.cut(df["score"], np.arange(min(df.score),max(df.score)+1,(max(df.score)-min(df.score))/n_bin),include_lowest=True))["score"].mean()
        else:
            SummaryTable["actual_pd"]=(df.groupby(pd.cut(df["score"], np.arange(min(df.score),max(df.score),(max(df.score)-min(df.score))/n_bin),include_lowest=True)).sum()["outcome"]/  
                          df.groupby(pd.cut(df["score"], np.arange(min(df.score),max(df.score),(max(df.score)-min(df.score))/n_bin),include_lowest=True)).size())
            SummaryTable["pred_pd"]=df.groupby(pd.cut(df["score"], np.arange(min(df.score),max(df.score),(max(df.score)-min(df.score))/n_bin),include_lowest=True))["score"].mean()

        if plot==True:
            SummaryTable.plot(kind="line",marker="o",legend=True,grid=True)
        return SummaryTable
    
    def CalcModelROC(raw_data,response_var,explanatory_var_1,explanatory_var_2="",explanatory_var_3="",explanatory_var_4="",explanatory_var_5="",explanatory_var_6="",explanatory_var_7="",explanatory_var_8="",explanatory_var_9="",explanatory_var_10=""):
        '''
        Directly generate ROC stats from a given model. 
        raw_data: pandas.DataFrame. The dataset contains all variables used in model generation.
        response_var: string. Column name of response var.
        explanatory_var_1:string. Column name of response var.
        explanatory_var_2-10:string.Optional. Column name of response var.
        '''
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        explanatory_var_list=[explanatory_var_1,explanatory_var_2,explanatory_var_3,explanatory_var_4,explanatory_var_5,explanatory_var_6,explanatory_var_7,explanatory_var_8,explanatory_var_9,explanatory_var_10]
        model_form=response_var+" ~ "+explanatory_var_1
        for i in range(9):
            if explanatory_var_list[i+1]!="":
                 model_form+= " + "+explanatory_var_list[i+1]
        model=sm.formula.glm(model_form,family=sm.families.Binomial(), data=raw_data).fit()
        raw_data["regression_result"]=model.params[0]+model.params[1]*raw_data[explanatory_var_1]
        for i in range(9):
            if explanatory_var_list[i+1]!="":
                raw_data["regression_result"]+=model.params[i+2]*raw_data[explanatory_var_list[i+1]]
        print(ValidationTool.CalcROCStats(raw_data["regression_result"],raw_data[response_var]))
        return raw_data
    
    def SummarizeROCStats(raw_data,default_flag_column,score,by_vars,continuous_var=False,plot=False,n_bin=10):
        '''
        Until 20190214, this function cannot perform well regarding continuous variable.
        '''
        '''
        Summarize ROC stats according to by_vars
        raw_data: pandas.DataFrame. dataset that contains all variables
        default_flag_column: String. column name of the default flag in dataset
        score:String. column name of the generated score in dataset
        by_vars: String. Column name of the groupby variable
        continuous_var: Boolean. Default as False. If select as True, function will bucket the variable to do groupby.
        n_bin: int. If contunuous_var is set as True, function will bucket the variable into "n_bin" bins
        plot: Boolean. Default as False. If select True, function will generate graph
        '''
        import numpy as np
        import pandas as pd
        data=raw_data[[score,by_vars,default_flag_column]]
        max_by_vars=max(data[by_vars])
        min_by_vars=min(data[by_vars])
        result=pd.DataFrame()
        result=ValidationTool.SummarizeDefault(raw_data=raw_data,by_vars=by_vars,continuous_var=continuous_var,
                                               default_flag_column=default_flag_column,n_bin=n_bin)
        result["C_stat"]=0
        result["KS"]=0

        if continuous_var==False:
            for i in list(result.index.values):
                result.loc[i,"C_stat"]=ValidationTool.CalcROCStats(default_flag_data=data[default_flag_column][data[by_vars]==i],
                                                                s=data[score][data[by_vars]==i])["c_stat"]
                result.loc[i,"KS"]=ValidationTool.CalcROCStats(default_flag_data=data[default_flag_column][data[by_vars]==i],
                                                                s=data[score][data[by_vars]==i])["ks_stat"]
                result.loc[i,"AR"]=ValidationTool.CalcROCStats(default_flag_data=data[default_flag_column][data[by_vars]==i],
                                                                s=data[score][data[by_vars]==i])["ar_stat"]
        else:
            if raw_data[by_vars].isnull().values.any()==True:
                group_by_range=(pd.cut(np.array(data[by_vars]), np.arange(min_by_vars,max_by_vars+1,
                                                                                          (max_by_vars-min_by_vars)/n_bin),include_lowest=True)
                                                         .add_categories("missing"))
                group_by_range=group_by_range.fillna("missing")
            else:
                group_by_range=(pd.cut(np.array(data[by_vars]), np.arange(min_by_vars,max_by_vars+1,(max_by_vars-min_by_vars)/n_bin),include_lowest=True))
            for i in list(result.index.values):
                g=data.groupby(group_by_range)
                result["C_stat"]=g.apply(ValidationTool.CalcROCStats(default_flag_data=g[default_flag_column],s=g[score]))["c_stat"]
                result.loc[i,"C_stat"]=ValidationTool.CalcROCStats(default_flag_data=data[default_flag_column][data["groupby_index"]==str(i)],
                                                                    s=data[score][data["groupby_index"]==str(i)])["c_stat"]
                result.loc[i,"KS"]=ValidationTool.CalcROCStats(default_flag_data=data[default_flag_column][data["groupby_index"]==str(i)],
                                                                    s=data[score][data["groupby_index"]==str(i)])["ks_stat"]
                result.loc[i,"AR"]=ValidationTool.CalcROCStats(default_flag_data=data[default_flag_column][data["groupby_index"]==str(i)],
                                                                    s=data[score][data["groupby_index"]==str(i)])["ar_stat"]
        if plot==True:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=3, ncols=1)
            result["C_stat"].plot(kind="line",marker="o",grid=True,ax=axes[0],title="C_Stat")
            result["KS"].plot(kind="line",marker="o",grid=True,ax=axes[1],title="KS")
            result["AR"].plot(kind="line",marker="o",grid=True,ax=axes[2],title="AR")
        return result

    
    def CalcWOE_IV(data,default_flag_column,categorical_column,event_list=[1,"Yes"],continuous_variable=False,n_bucket=10):
        '''
        Calculate IV of each variable
        Calculation refer to: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        data: pandas.DataFrame. dataset that contains all variables
        default_flag_column: String. column name of the default flag in dataset
        categorical_column:String. column name of the categorical column
        continuous_var: Boolean. Default as False. If select as True, function will bucket the variable to do groupby.
        n_bin: int. If contunuous_var is set as True, function will bucket the variable into "n_bin" bins
        event_list: list. The list contains all possibility than can be considered as "event"
        '''
        import numpy as np
        import pandas as pd
        used_data=pd.DataFrame()
        used_data[default_flag_column]=data[default_flag_column]
        used_data[categorical_column]=data[categorical_column]
        used_data=used_data.dropna(subset=[default_flag_column])
        used_data["event_flag"]="nonevent" 
        used_data.loc[used_data[default_flag_column].isin(event_list),"event_flag"]="event"
        used_data["event_flag_num"]=0
        used_data.loc[used_data["event_flag"]=="event","event_flag_num"]=1
        used_data["nonevent_flag_num"]=0
        used_data.loc[used_data["event_flag"]=="nonevent","nonevent_flag_num"]=1
        if np.issubdtype(used_data[categorical_column].dtype,np.number)==True:
            group_by_range=(pd.cut(np.array(used_data[categorical_column]),np.arange(min(used_data[categorical_column]),max(used_data[categorical_column])+1,(max(used_data[categorical_column])-min(used_data[categorical_column]))/n_bucket),include_lowest=True).add_categories("missing"))
            group_by_range=group_by_range.fillna("missing")
        else:
            group_by_range=used_data[categorical_column].unique()
        
        if continuous_variable==False:
            used_data["group_size"]=used_data.groupby(categorical_column)["event_flag_num"].transform("size")
            used_data["event_in_group"]=used_data.groupby(categorical_column)["event_flag_num"].transform("sum")
            used_data["nonevent_in_group"]=used_data.groupby(categorical_column)["nonevent_flag_num"].transform("sum")
        else:
            used_data["group_size"]=(used_data.groupby
                                     (group_by_range)
                                     ["event_flag_num"].transform("size"))
            used_data["event_in_group"]=(used_data.groupby
                                         (group_by_range)
                                        ["event_flag_num"].transform("sum"))
            used_data["nonevent_in_group"]=(used_data.groupby
                                         (group_by_range)
                                        ["nonevent_flag_num"].transform("sum"))

        
        used_data["modified_event_flag_num"]=used_data["event_flag_num"].apply(lambda x : x+0 if x<=0 else x)
        used_data["modified_nonevent_flag_num"]=used_data["nonevent_flag_num"].apply(lambda x : x+0 if x<=0 else x)
        used_data["weight_event"]=used_data["modified_event_flag_num"]/used_data["modified_event_flag_num"].sum()
        used_data["weight_nonevent"]=used_data["modified_nonevent_flag_num"]/used_data["modified_nonevent_flag_num"].sum()

        WOE_table=pd.DataFrame()
        if continuous_variable==False:
            WOE_table["weight_group"]=used_data.groupby(categorical_column).size()/used_data.shape[0]        
            WOE_table["event_num"]=used_data.groupby(categorical_column)["event_flag_num"].sum()
            WOE_table["nonevent_num"]=used_data.groupby(categorical_column)["nonevent_flag_num"].sum()
            WOE_table["weight_event"]=used_data.groupby(categorical_column)["weight_event"].sum().apply(lambda x : x+0.5 if x<=0 else x)
            WOE_table["weight_nonevent"]=used_data.groupby(categorical_column)["weight_nonevent"].sum().apply(lambda x : x+0.5 if x<=0 else x)
        else:
            WOE_table["weight_group"]=(used_data.groupby
                                       (group_by_range).size()/used_data.shape[0])  
            WOE_table["event_num"]=(used_data.groupby
                                    (group_by_range)["event_flag_num"].sum())
            WOE_table["nonevent_num"]=(used_data.groupby
                                       (group_by_range)
                                       ["nonevent_flag_num"].sum())
            WOE_table["weight_event"]=(used_data.groupby
                                       (group_by_range)
                                       ["weight_event"].sum().apply(lambda x : x+0.5 if x<=0 else x))
            WOE_table["weight_nonevent"]=(used_data.groupby
                                          (group_by_range)
                                          ["weight_nonevent"].sum().apply(lambda x : x+0.5 if x<=0 else x))

        WOE_table["WOE"]=np.log(WOE_table["weight_event"])-np.log(WOE_table["weight_nonevent"])
        WOE_table["IV"]=WOE_table["WOE"]*(WOE_table["weight_event"]-WOE_table["weight_nonevent"])
        if continuous_variable==False:
            WOE_table["pd"]=(used_data.groupby(categorical_column)["event_flag_num"].sum()
                             /used_data.groupby(categorical_column).size())
        else:
            WOE_table["pd"]=(used_data.groupby
                             (group_by_range)
                             ["event_flag_num"].sum()
                             /used_data.groupby
                             (group_by_range)
                             .size())
        WOE_table["log_odds"]=np.log(WOE_table["pd"]/(1-WOE_table["pd"]))
        if WOE_table.weight_event.sum()-1<0.0001 and WOE_table.weight_nonevent.sum()-1<0.0001:
            print("Completed!")
        else:
            print("Please check: weight added not to 1.")
        print("{a}'s WOE:{b:4.4f},IV:{c:4.4f}.".format(a=categorical_column,b=WOE_table.WOE.sum(),c=WOE_table.IV[WOE_table.IV<1].sum()))
        return [WOE_table.IV[WOE_table.IV<1].sum(),WOE_table]

    def PerformRandomCV(data,response_var,explanatory_var_list=[],k=20,plot=True,sampling_as_default_percentage=False,RegType="Logit"):
        '''
        Do Random Cross Validation of model to check model stability. 
        Return standard deviation of each coefficients.
        raw_data: pandas.DataFrame. The dataset contains all variables used in model generation.
        response_var: string. Column name of response var.
        explanatory_var_list: list. List of all names of explanatory variables
        k: int. 1/k samples will be eliminated as train set.
        plot: Boolean. Default as True. Plot std. dev of each coefficient.
        sampling_as_default_percentage: Boolean. Default as False. If selected as True, sampling will be done as the default rate.
        
        return std.dev of coefficients during k-time regression
        '''
        import statsmodels.api as sm
        import sys
        import time, sys
        import pandas as pd
        import numpy as np
        import statsmodels.formula.api as smf
        explanatory_var_list=explanatory_var_list
        model_form=response_var+" ~ "+explanatory_var_list[0]
        n_var=1
        dict_params={}
        R_square_adj={}
        for i in range(len(explanatory_var_list)-1):
            if explanatory_var_list[i+1]!="":
                model_form+= " + "+explanatory_var_list[i+1]
                n_var+=1
        for i in range(k):
            time.sleep(0.1)
            ValidationTool.update_progress("Running Random Sampling", (i+1)/k)
            if sampling_as_default_percentage==False:
                used_data=data.sample(frac=1-1/k)
            else:
                if data[response_var].isin([0,1]).all():
                    used_data_1=data[data[response_var]==1].sample(frac=1-1/k)
                    used_data_2=data[data[response_var]==0].sample(frac=1-1/k)
                    used_data=pd.concat([used_data_1,used_data_2])
                else:
                    print ("The response variable is not a boolean variable.\n Set input of sampling_as_default_percentage as False to continue.")
                    return
            if RegType=="Logit":
                model=sm.formula.glm(model_form,family=sm.families.Binomial(), data=used_data).fit()
            elif RegType=="OLS":
                formula = "{} ~ {} ".format(response_var,' + '.join(explanatory_var_list))
                model=smf.ols(formula, used_data).fit()
                R_square_adj[i+1]=model.rsquared_adj

            dict_params[i+1]=model.params
            
        df_params=pd.DataFrame(dict_params)
#         if RegType=="OLS":
#             df_R_squared_adj=pd.DataFrame(R_square_adj)
        if plot==True:
            df_params.std(axis=1).plot(kind="bar",grid=True,title="Standard Deviation of Each Parameter")
        return [df_params,R_square_adj]
    
    def PerformK_Fold(data,response_var, explanatory_var_list=[],k=10,RegType="Logit"):
        '''
        Perform k-fold cross validation
        data: pandas.DataFrame. dataset that contains all variables.
        response_var: string. column name of response variable
        explanatory_var_list: list of strings. list that contains all explanatory variables
        '''
        from sklearn.model_selection import cross_val_score
        from sklearn import datasets, linear_model
        import pandas as pd
        import numpy as np
        y = data[response_var]
        X = data[explanatory_var_list]
        if RegType=="Logit":
            Regression = linear_model.LogisticRegression()
        elif RegType=="OLS":
            Regression = linear_model.LinearRegression()
        return cross_val_score(Regression, X, y, cv=k)
    
    def fit_logistic_reg(X,Y):
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pandas as pd
        import numpy as np
        model_k = linear_model.LogisticRegression(fit_intercept = True)
        model_k.fit(X,Y)
        RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
        R_squared = model_k.score(X,Y)
        return RSS, R_squared
    
    def BestSubsetSelection(data,response_var,explanatory_var_list=[],plot=True):
        '''
        Run best subset selection
        data: pandas.DataFrame. raw dataset
        response_var: string. Column name of response var
        explanatory_var_list: list of string. Names of columns of all explanatory candidates.

        return all combinations' RSS and R_squared
        '''
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pandas as pd
        import time, sys
        import itertools
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        i=0
        k=len(explanatory_var_list)
        RSS_list,R_squared_list,feature_list,numb_features,AIC_list,BIC_list=[],[],[],[],[],[]
        for m in range(1,k+1):
            for combo in itertools.combinations(explanatory_var_list,m):
                temp_combo=list(combo)
                for var in combo:
                    if np.issubdtype(data[var].dtype,np.number)!=True:
                        temp_combo.remove(var)
                        unique_value_list=data[var].unique()
                        for new_column in unique_value_list[:-1]:
                            data["dummy_"+new_column]=0
                            data.loc[data[var] == new_column, "dummy_"+new_column] = 1
                            temp_combo.append("dummy_"+new_column)
                X=data[temp_combo]
                Y=data[response_var]
                temp=ValidationTool.fit_logistic_reg(X,Y)
                RSS_list.append(temp[0])
                R_squared_list.append(temp[1])
                feature_list.append(combo)
                numb_features.append(len(combo))
                i+=1
                ValidationTool.update_progress(job_title="Running Best Subset Selection",progress=(i+1)/(2**k))
                time.sleep(0.1)
        df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
        df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
        df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
        df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
        df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
        if plot==True:
            fig = plt.figure(figsize = (16,6))
            ax = fig.add_subplot(1, 2, 1)

            ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
            ax.set_xlabel('# Features')
            ax.set_ylabel('RSS')
            ax.set_title('RSS - Best subset selection')
            ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
            ax.legend()

            ax = fig.add_subplot(1, 2, 2)
            ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
            ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
            ax.set_xlabel('# Features')
            ax.set_ylabel('R squared')
            ax.set_title('R_squared - Best subset selection')
            ax.legend()

            plt.show()
        return df_min,df_max
    
    def ForwardStepwiseSelection(data,response_var,explanatory_var_list=[],plot=True):
        '''
        Max suppoted number of Vars is 11 due to max recursion limit will breacch for vars more than 12
        Do forward stepwise selection
        data: pandas.DataFrame. raw dataset
        response_var: string. Column name of response var
        explanatory_var_list: list of string. Names of columns of all explanatory candidates.
        '''
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import pandas as pd
        import time, sys
        sys.setrecursionlimit(10**6) 
        import itertools
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        #X=data[explanatory_var_list]
        #Y=data[response_var]
        k=len(explanatory_var_list)

#        remaining_features = list(X.columns.values)
#        features , features_show= [],[]
#        RSS_list, R_squared_list = [np.inf], [np.inf] 
#        features_list ={}
        aic_list,bic_list=[],[]
        var_list=[]
        RSS=[]
        i=0
        formula=response_var+"~"
        for var in explanatory_var_list:
            i=i+1
            ValidationTool.update_progress(job_title="Running Forward Stepwise Selection:",progress=i/k)
            time.sleep(0.1)
            if var!=explanatory_var_list[0]:
                if np.issubdtype(data[var].dtype,np.number)!=True:
                    formula=formula+"+C("+var+")"
                else:
                    formula=formula+"+"+var
            elif var==explanatory_var_list[0]:
                if np.issubdtype(data[var].dtype,np.number)!=True:
                    formula=formula+"C("+var+")"
                else:
                    formula=formula+var
            model=sm.formula.glm(formula,family=sm.families.Binomial(), data=data).fit()
            var_list.append(var)
            aic_list.append(model.aic)
            bic_list.append(model.bic)
        res=pd.DataFrame()
        res["Var"]=var_list
        res["AIC"]=aic_list
        res["BIC"]=bic_list
        
#         for i in range(1,k+1):
#             ValidationTool.update_progress(job_title="Running Forward Stepwise Selection:",progress=i/k)
#             time.sleep(0.1)
#             best_RSS = np.inf

#             for combo in itertools.combinations(remaining_features,1):
                
#                 for var in combo: # This is a one-time iteration
#                     temp_combo=list(combo) # List that contains only one element
#                     if np.issubdtype(data[var].dtype,np.number)!=True:
                        
#                         temp_combo.remove(var) # Return an emtpy list
#                         unique_value_list=data[var].unique()
#                         for new_column in unique_value_list[:-1]:
#                             data["dummy_"+new_column]=0
#                             data.loc[data[var] == new_column, "dummy_"+new_column] = 1
#                             temp_combo.append("dummy_"+new_column)
#                 temp_combo.extend(features)
#                 X=data[temp_combo]
#                 Y=data[response_var]
#                 RSS=ValidationTool.fit_logistic_reg(X,Y) 

#                 if RSS[0] < best_RSS:
#                     best_RSS = RSS[0]
#                     best_R_squared = RSS[1]
#                     best_feature = combo[0]
#                     dummy_best_feature=temp_combo # To record all dummy variables generated in this step

#             #Updating variables for next loop
#             features.extend(dummy_best_feature) # For regression Purpose
#             features_show.append(best_feature) # For show purpose
#             remaining_features.remove(best_feature)

#             #Saving values for plotting
#             RSS_list.append(best_RSS)
#             R_squared_list.append(best_R_squared)
#             features_list[i] = features_show.copy()
#         df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
#         df1['numb_features'] = df1.index
#         #Initializing useful variables
#         m = len(Y)
#         p = k
#         hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])

#         #Computing
#         df1['C_p'] = (1/m) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
#         df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
#         df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['numb_features'] * hat_sigma_squared )
#         df1['R_squared_adj'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['numb_features'] -1))
        
#         if plot==True:
#             import matplotlib.pyplot as plt
#             variables = ['C_p', 'AIC','BIC','R_squared_adj']
#             fig = plt.figure(figsize = (18,6))

#             for i,v in enumerate(variables):
#                 ax = fig.add_subplot(1, 4, i+1)
#                 ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
#                 ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
#                 if v == 'R_squared_adj':
#                     ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
#                 else:
#                     ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
#                 ax.set_xlabel('Number of predictors')
#                 ax.set_ylabel(v)

#             fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
#             plt.show()
        return res
    
        
    def CalcSpearmanCorrelation(data,var_1,var_2):
        '''
        Calculate spearman Correlation
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        '''
        import scipy.stats as stats
        import pandas as pd
        import numpy as np
        pho,p_value=stats.spearmanr(data[var_1],data[var_2])
        return pho
    
    def CalcKendallTau(data,var_1,var_2):
        '''
        Calculate KendallTau
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        '''
        import scipy.stats as stats
        import pandas as pd
        import numpy as np
        tau, p_value = stats.kendalltau(data[var_1], data[var_2])
        return tau
    
    def CalcSamplingGoodmanKruskalGamma(data,var_1, var_2,n_sample=1000):
        '''
        Calculate GoodmanKruskalGamma estimator with sampling data 
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        '''
        import sys
        import time, sys
        import pandas as pd
        import numpy as np
        concordance=0
        discordance=0
        x_1=list(data[var_1].sample(n=n_sample,random_state=1))
        x_2=list(data[var_2].sample(n=n_sample,random_state=1))
        for i in range(len(x_1)):
            time.sleep(0.1)
            ValidationTool.update_progress(job_title="Calculation Goodman Kruskal Gamma",progress=(i+1)/(len(x_1)))
            for j in range(len(x_2)):
                x_1_dir = x_1[i] - x_1[j]
                x_2_dir = x_2[i] - x_2[j]
                sign = x_1_dir * x_2_dir
                if sign > 0:
                    concordance+=1
                elif sign < 0:
                    discordance+=1
        return (concordance-discordance)/(concordance+discordance)
      
        
    def CalcSomersD(data,var_1,var_2):
        '''
        Calculate Somers' D
        data: pandas.DataFrame. Contains var_1 and var_2
        var_1,var_2: string. column name of var_1 and var_2
        '''
        result=ValidationTool.CalcROCStats(data[var_1],data[var_2])
        return result["c_stat"]
    
    def GenerateRankCorrelation(data,var_1,var_2):
        return ({"SpearmanCorrelation":ValidationTool.CalcSpearmanCorrelation(data,var_1,var_2),
                 "Kendall_Tau":ValidationTool.CalcKendallTau(data,var_1,var_2),
                 "GoodmanKruskal_Gamma":ValidationTool.CalcSamplingGoodmanKruskalGamma(data,var_1, var_2,n_sample=100),
                 "Somers' D":ValidationTool.CalcSomersD(data,var_1,var_2)})
    def OLSForwardStepwise(data,response_var,explanatory_var_list=[],plot=True):

        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import pandas as pd
        import time, sys
        sys.setrecursionlimit(10**6) 
        import itertools
        import statsmodels.formula.api as smf
        X=data[explanatory_var_list]
        Y=data[response_var]
        k=len(explanatory_var_list)

        remaining_features = list(X.columns.values)
        features , features_show= [],[]
        RSS_list, R_squared_list,AIC_list= [np.inf], [np.inf] , [np.inf] 
        features_list ={}
        AIC_list,BIC_list=[],[]
        RSS=[]
        
        for i in range(1,k+1):
            ValidationTool.update_progress(job_title="Running Forward Stepwise Selection:",progress=i/k)
            time.sleep(0.1)
            best_RSS = np.inf

            for combo in itertools.combinations(remaining_features,1):
                
                for var in combo: # This is a one-time iteration
                    temp_combo=list(combo) # List that contains only one element
                    if np.issubdtype(data[var].dtype,np.number)!=True:
                        temp_combo.remove(var) # Return an emtpy list
                        unique_value_list=data[var].unique()
                        for new_column in unique_value_list[:-1]:
                            data["dummy_"+new_column]=0
                            data.loc[data[var] == new_column, "dummy_"+new_column] = 1
                            temp_combo.append("dummy_"+new_column)
                temp_combo.extend(features)
                X=data[temp_combo]
                Y=data[response_var]
                formula = "{} ~ {} ".format(response_var,' + '.join(temp_combo))
                res=smf.ols(formula, data).fit()

                RSS = [res.mse_total,res.rsquared_adj,res.aic]

                if RSS[0] < best_RSS:
                    best_RSS = RSS[0]
                    best_R_squared = RSS[1]
                    best_AIC=RSS[2]
                    best_feature = combo[0]
                    dummy_best_feature=temp_combo # To record all dummy variables generated in this step

            #Updating variables for next loop
            features.extend(dummy_best_feature) # For regression Purpose
            features_show.append(best_feature) # For show purpose
            remaining_features.remove(best_feature)

            #Saving values for plotting
            RSS_list.append(best_RSS)
            R_squared_list.append(best_R_squared)
            AIC_list.append(best_AIC)
            features_list[i] = features_show.copy()
#            
        df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list })], axis=1, join='inner')
        df1["AIC"]=AIC_list
        df1['numb_features'] = df1.index
        if plot==True:
            import matplotlib.pyplot as plt
            variables = ['AIC','R_squared']
            fig = plt.figure(figsize = (18,6))

            for i,v in enumerate(variables):
                ax = fig.add_subplot(1, 2, i+1)
                ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
                ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
                if v == 'R_squared':
                    ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
                else:
                    ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
                ax.set_xlabel('Number of predictors')
                ax.set_ylabel(v)

            fig.suptitle('Subset selection using AIC, Adjusted R2', fontsize = 16)
            plt.show()
        return df1