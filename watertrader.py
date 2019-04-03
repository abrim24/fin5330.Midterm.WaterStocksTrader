"""
TraderEnv class
author: andy
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd

class TraderEnv(gym.Env):

    def __init__(self):
        #trading params
        self.length = 0.5 # actually half the pole's length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.rnd = 2
        self.pairs=[["CTWS","AWR"]]#WTR
        self.lags = 5
        self.dataidx = self.lags
        self.df = pd.read_csv("waterStocksWithReturns.csv")#Stocks_6yr.csv")
        self.dflen = len(self.df)
        self.training = True
        self.trainIters = 0
        self.trainMax = 500 #should match TRAINING_ITERS in learning algorithm
        self.testIdx = 2400#water last year 214 #idx in dataset where test data begins.  i.e. first 5 years training data, last year test data
        self.firstEp = True
        
        #env params
        self.action_space = spaces.Discrete(3)# [-1,0,1] #actions are: -1 short, 0 no position, 1 long
        self.high = np.array([3.0, 3.0, 0.0, 0.0]) #max: returns=3.0 (300%), sp/spmn=2.0(usually near 1.0)
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)
        self.seed()
        
        
        #building environment states for each pair
        j = 0
        for p in self.pairs:
            
            self.df["sp"+str(j)] = np.abs( self.df[p[0]] - self.df[p[1]] )
            
            self.df["sprets"+str(j)] = pd.Series([float("inf") for x in range(len(self.df["sp"+str(j)]))])
            self.df["spmn"+str(j)] = pd.Series([float("inf") for x in range(len(self.df["sp"+str(j)]))])
            self.df["sp_spmn"+str(j)] = pd.Series([float("inf") for x in range(len(self.df["sp"+str(j)]))])
            
            #TODO: try higher precision round, 4 not 2
            #calc spread returns
            for i in range(len(self.df["sp"+str(j)])-1): 
                self.df["sprets"+str(j)][i+1] = np.round( ((self.df["sp"+str(j)][i+1] - self.df["sp"+str(j)][i])/self.df["sp"+str(j)][i]),self.rnd)
            
            #calc srets mean
            for i in range(len(self.df["sp"+str(j)])-self.lags): 
                self.df["spmn"+str(j)][self.lags+i] = np.round( (np.mean(self.df["sprets"+str(j)][i:i+self.lags])), self.rnd)
            
            self.df["sp_spmn"+str(j)] = np.round(self.df["sprets"+str(j)] / self.df["spmn"+str(j)],self.rnd)
            
            print("dataframe: ",self.df[[p[0],p[1],"sp"+str(j),"sprets"+str(j),"spmn"+str(j),"sp_spmn"+str(j)]])
        
            j += 1
            
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        # print("new action: ",action)
        self.dataidx += 1
        idx = self.dataidx
        
        sprets0 = self.df["sprets0"][self.dataidx]
        spmn0 = self.df["spmn0"]
        sp_spmn0 = self.df["sp_spmn0"][self.dataidx]
        
        # sprets1 = self.df["sprets1"][self.dataidx]
        # sp_spmn1 = self.df["sp_spmn1"][self.dataidx]
        
        # print("sprets: ",sprets)
        # print("sp_spmn: ",sp_spmn)
        rewards = sprets0 * (action-1)#action comes as index 0,1,2 converting to position -1,0,1
        if self.firstEp: print("sprets0, rewards: ",sprets0,rewards)
        # print("self.dataidx: ",self.dataidx)
        #return np.array((3.0,3.0,3.0,3.0)),1.0,False,{}
        
        # 
        done = False
        if self.training:
            done = self.dataidx == self.testIdx-1 #finished training, before we reach test data
        else:
            done = self.dataidx == self.dflen-1 #finished testing at end of data set
            
        if done and self.training: 
            self.trainIters += 1
            self.firstEp = False
            # print("self.trainIters: ",self.trainIters)
        if self.trainIters == self.trainMax: 
            self.training = False
            
        return np.array((sprets0,sp_spmn0,0.0,0.0)), rewards, done, {}
        
        
    def reset(self):
        if self.training:
            self.dataidx = self.lags
            return np.array((self.df["sp"+str(0)][1],0.0,0.0,0.0))
        else:
            self.dataidx = self.testIdx
            return np.array((self.df["sp"+str(0)][self.testIdx],0.0,0.0,0.0))
            