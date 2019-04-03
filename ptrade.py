import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import numpy as np
import os

DEBUG = False

class WaterPairsTrader():
    def __init__(self):
        self.lags = 10
        self.rnd = 3
        self.completeData = "data/waterStocksWithReturns.csv"
        self.preData = "data/WaterStocksAdj.csv"
        if not os.path.isfile(self.completeData):
            print("Complete file not found, generating data file...")
            
            self.df = pd.read_csv("data/WaterStocksAdj.csv")
            #self.df = self.df.head(30)
            
            self.df.columns = ['date','CTWS', 'WTR', 'AWR']
            #calcing spreads
            self.df["ca_sp"] = np.round(np.abs(self.df["CTWS"]-self.df["AWR"]),self.rnd)
            self.df["cw_sp"] = np.round(np.abs(self.df["CTWS"]-self.df["WTR"]),self.rnd)
            self.df["aw_sp"] = np.round(np.abs(self.df["AWR"]-self.df["WTR"]),self.rnd)
            #init spread mean
            self.df["ca_spmn"] = pd.Series([float("inf") for x in range(len(self.df))])
            self.df["cw_spmn"] = pd.Series([float("inf") for x in range(len(self.df))])
            self.df["aw_spmn"] = pd.Series([float("inf") for x in range(len(self.df))])
            
            #calc spread mean
            for i in range(len(self.df)-self.lags): 
                self.df["ca_spmn"][i+self.lags] = np.round(np.mean(self.df["ca_sp"][i:i+self.lags]),self.rnd)
                self.df["cw_spmn"][i+self.lags] = np.round(np.mean(self.df["cw_sp"][i:i+self.lags]),self.rnd)
                self.df["aw_spmn"][i+self.lags] = np.round(np.mean(self.df["aw_sp"][i:i+self.lags]),self.rnd)
                
            #init spread returns
            self.df["ca_sprets"] = pd.Series([float("inf") for x in range(len(self.df))])
            self.df["cw_sprets"] = pd.Series([float("inf") for x in range(len(self.df))])
            self.df["aw_sprets"] = pd.Series([float("inf") for x in range(len(self.df))])
            
            #calc spread returns
            for i in range(len(self.df)-1):
                self.df["ca_sprets"][i+1] = np.round( ((self.df["ca_sp"][i+1] - self.df["ca_sp"][i])/self.df["ca_sp"][i]),self.rnd)
                self.df["cw_sprets"][i+1] = np.round( ((self.df["cw_sp"][i+1] - self.df["cw_sp"][i])/self.df["cw_sp"][i]),self.rnd)
                self.df["aw_sprets"][i+1] = np.round( ((self.df["aw_sp"][i+1] - self.df["aw_sp"][i])/self.df["aw_sp"][i]),self.rnd)
                
            #init spread returns mean
            self.df["ca_spretsmn"] = pd.Series([float("inf") for x in range(len(self.df))])
            self.df["cw_spretsmn"] = pd.Series([float("inf") for x in range(len(self.df))])
            self.df["aw_spretsmn"] = pd.Series([float("inf") for x in range(len(self.df))])
            
            #calc spread returns mean
            for i in range(len(self.df)-self.lags): 
                self.df["ca_spretsmn"][i+self.lags] = np.round(np.mean(self.df["ca_sprets"][i:i+self.lags]),self.rnd)
                self.df["cw_spretsmn"][i+self.lags] = np.round(np.mean(self.df["cw_sprets"][i:i+self.lags]),self.rnd)
                self.df["aw_spretsmn"][i+self.lags] = np.round(np.mean(self.df["aw_sprets"][i:i+self.lags]),self.rnd)
                
            self.df.to_csv(self.completeData)
            if DEBUG: print("df...\n ",self.df.head(20))
        else:
            print("Complete data file exists, loading data...")
            self.df = pd.read_csv(self.completeData)
            
    def testCoint(self):
        #pairs ca (CTWS/AWR), cw (CTWS/WTR), aw (AWR/WTR)
        df = self.df
        rnd = self.rnd
        
        #creating dictionaries to store results
        dkeys = ["corr", "pval", "adfstat", "conf1", "conf5", "conf10"]
        ca=dict()
        ca2=dict()
        cw=dict()
        cw2=dict()
        aw=dict()
        aw2=dict()
        dicts = [ca, ca2, cw, cw2, aw, aw2]
        for d in dicts:
            for k in dkeys:
                d[k] = list()
        
        pairs = [["CTWS","AWR"],["AWR","CTWS"],["CTWS","WTR"],["WTR","CTWS"],["AWR","WTR"],["WTR","AWR"]]
        for i in range(12):
            for j in range(len(dicts)):
                dRes = sm.OLS(df[pairs[j][0]][(i*253):((i+1)*253)],df[pairs[j][1]][(i*253):((i+1)*253)]).fit()
                dAdf = ts.adfuller(dRes.resid)
                dicts[j]["corr"].append(np.round(np.corrcoef(df[pairs[j][0]][(i*253):((i+1)*253)],df[pairs[j][1]][(i*253):((i+1)*253)])[1,0],rnd))
                dicts[j]["adfstat"].append(np.round(dAdf[0],rnd))
                dicts[j]["pval"].append(np.round(dAdf[1],rnd))
                dicts[j]["conf1"].append(np.round(dAdf[4]['1%'],rnd))
                dicts[j]["conf5"].append(np.round(dAdf[4]['5%'],rnd))
                dicts[j]["conf10"].append(np.round(dAdf[4]['10%'],rnd))
                
        for i in range(len(dicts)):
            print("Results for pair: ",pairs[i],"--------------")
            for j in dkeys:
                print(j,":",dicts[i][j])
    
    #spread trading 
    #calculates returns slightly differently than tradeReturns below
    #tradePrices begins with a starting amount, and calculates returns against ending amount
    def tradePrices(self):
        taus = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.25, 0.50, 0.75]
        spds = ["ca_sp","cw_sp","aw_sp"]
        for spKey in spds:
            for tau in taus:
                buy = sell = profit = 0.0
                start = float("inf")
                returns = []
                sp = self.df[spKey]
                spmn = self.df[spKey+"mn"]
                totalProfit = 0.0
                #print("Running strategy for spread ",spKey,"tau: ",tau)
                for i in range(self.lags,len(self.df)):
                    #print(i,"sp[i],spmn[i]: ",sp[i],spmn[i],sp[i]/spmn[i])
                    if sp[i] < spmn[i] * (1-tau) and buy == 0.0:#buy spread
                        buy = sp[i]
                        sell = 0.0
                        if start == float("inf"): start = buy
                        #print("buying spread at price: ",buy)
                    elif sp[i] > spmn[i] * (1+tau) and buy != 0.0:
                        sell = sp[i]
                        profit += sell - buy
                        buy = 0.0
                        #print("selling spread at price: ",sell," profit: ",profit)
                    else:
                        pass
                    if i % 253 == 0:
                        #print("year",i//253,"tau",tau,"profit: ",profit," returns: ",profit/start)
                        returns.append(profit/start)
                        totalProfit += profit
                        buy = sell = profit = 0.0
                        start = float("inf")
                print("**** Prices 12 year average annual returns for tau ", tau, ": ",np.mean(returns),"******")
        
    def tradeReturns(self):
        taus = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.25, 0.50, 0.75]
        spds = ["ca_sprets","cw_sprets","aw_sprets"]
        for spKey in spds:
            for tau in taus:
                cummReturns = 0.0
                yearlyReturns = []
                sp = self.df[spKey]
                spmn = self.df[spKey+"mn"]
                signal = 0
                #print("Running strategy for spread ",spKey,"tau: ",tau)
                for i in range(self.lags+1,len(self.df)):
                    returns = np.round(sp[i] * signal,4)
                    cummReturns += returns
                    #print(i,"sp[i],signal,returns,cummReturns: ",sp[i],signal,returns,cummReturns)
                    
                    if sp[i] < -tau:#long spread
                        signal = 1
                        #print("buying spread at price: ",sp[i])
                    elif sp[i] > tau:#short spread
                        signal = -1
                        #print("selling spread at price: ",sp[i])
                    else:
                        pass
                    
                    if i % 253 == 0:
                        #print("****** year",i//253,"returns: ",cummReturns,"******")
                        yearlyReturns.append(cummReturns)
                        cummReturns = 0.0
                        signal = 0
                        
                print("**** Returns 12 year average annual returns for spread returns",spKey,"tau ", tau, ": ",np.mean(yearlyReturns),"******")
                
    
def main():
    wpt = WaterPairsTrader()
    #wpt.testCoint()
    wpt.tradePrices()
    wpt.tradeReturns()
    
main()