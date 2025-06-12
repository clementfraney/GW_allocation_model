# -*- coding: utf-8 -*-
"""
@author: Clément Franey
clefraney@gmail.com

Updated: June 11 2025


Instructions:
    
Download the folder 'GW_allocation_model' anywhere in your computer.

Set the path of the folder where the python model is located in the variable "path_Model_folder".
Set the path of the folder where the solver is located in the variable "solverpath_exe".
Set the name of the sover in the variable "solvername".

You can modify the global parameters as you wish.

"""

#%% Import libraries

from pyomo.environ import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pickle

#%% Set-up the correct path

# =============================================================================
# Insert the path to the folder GW_allocation_model below
# =============================================================================

path_Model_folder = r'F:\Data\s232484\GW_allocation_model' # CHANGE TO YOUR PATH

# =============================================================================
# Insert the path to the Solver below
# =============================================================================

solvername='cplex'  # 'glpk' 'cplex'
# solverpath_exe = r'F:\Data\s232484\winglpk-4.65\glpk-4.65\\w64\\glpsol'
solverpath_exe = r'C:/Program Files/IBM/ILOG/CPLEX_Studio2212/cplex/bin/x64_win64/cplex'

# =============================================================================
# Savepath
# =============================================================================

savepath = path_Model_folder + r'/Results' 

#%% Set global parameter

# =============================================================================
# Global parameters of the simulation
# =============================================================================

Catch='Watersheds'   # name differently if using different catchents
Geo='Capital'  # name differently if using different catchents
error_func='NSE'     # NSE, RMSE or r2 (the error function used for parametrizing the GW linear reservoir model)
weeks = 1721 # Timesteps, maximum 1749 weeks  # 10 years = 521 weeks, 20 years = 1043 weeks, 30 years = 1565 weeks, 33 years = 1721 weeks

# =============================================================================
# Scenarii
# =============================================================================

scenario = 1  # 0 = Baseline , 1 = Maximum abstraction capacity 2 , 2 = local water exchanges

ind_2 = False
if scenario == 1:
    ind_2 = True


#%% Load the Data

ntimes = np.arange(1, weeks+1, 1)   # weekly timesteps
Catch_Geo = Catch + '_' + Geo

print('    ', Catch, Geo, 'for', weeks, 'weeks, scenario', scenario)
start = time.time()
print(time.strftime("%H:%M:%S") + ' Importing data...')

# =============================================================================
# Open the tables
# =============================================================================

os.chdir(path_Model_folder + '/Input data model')

Catchments=pd.read_csv('Table_Catchments_'+Catch_Geo+'.csv')
WF=pd.read_csv('Table_WF_'+Catch_Geo+'.csv')
WW=pd.read_csv('Table_WW_'+Catch_Geo+'.csv')
WSA=pd.read_csv('Table_WSA_'+Catch_Geo+'.csv')
Water_Transfer=pd.read_csv('Table_Water_Transfer_'+Catch_Geo+'.csv')
WTP=pd.read_csv('Table_WTP_'+Catch_Geo+'.csv')
WW_WSA=pd.read_csv('Matrix_WW_WSA_'+Catch_Geo+'.csv')
WF_WW=pd.read_csv('Matrix_WF_WW_'+Catch_Geo+'.csv')

if scenario == 2:
    WW_municipality = pd.read_excel('Anlaegid_hieraki.xlsx')


# =============================================================================
# Open the Water balance data 
# =============================================================================

os.chdir(path_Model_folder + '/Input data model/WB_' + Catch_Geo)

ncatch=np.array(Catchments['Catch_ID'])
K_optim=pd.read_csv('K_optim_'+error_func+'.csv')

inflow = np.empty((len(ncatch), len(ntimes))) # 1749 weeks of data, approx 33 years
WB_data=[]
for i in range(1,len(ncatch)+1):
    data=pd.read_csv('WB_SZ_'+Catch+'_'+str(int(i))+'.csv')
    WB_data.append(data)
    inflow[i-1,:] = data['MIKE SHE GW recharge (mm)'][:len(ntimes)]/1000*Catchments['Area (m2)'][i-1]/1000   # from mm to 10^3 m3
    
    #remove the negative data in the inflow ! 
    for t in range(1,len(ntimes)):
        if inflow[i-1,t-1]<0:
            inflow[i-1,t] += inflow[i-1,t-1]
            inflow[i-1,t-1] = 0
    if inflow[i-1,len(ntimes)-1]<0:
        inflow[i-1,len(ntimes)-1] = 0   

#%% Process the data

print(time.strftime("%H:%M:%S") + ' Processing the data...')

# =============================================================================
# Variables (ntimes, nwsa, ncatch)
# =============================================================================

# ncatch=np.array(ncatch['ID'])
ncatch=np.array(Catchments['Catch_ID'])
#ntimes already defined
nyear=np.arange(1, int(len(ntimes)//52.18)+2, 1) # yearly index 
nwsa = np.array(WSA['WSAID'])
nww = np.array(WW['WWID'])
nwf = np.array(WF['WFID'])

# =============================================================================
# Convert the WTP in weekly time steps
# =============================================================================

WTP_weekly=pd.DataFrame(columns=['Time step','WTP_HH','WTP_Ind','WTP_PS','WTP_Agri'])
WTP_weekly['Time step']=data['Start time of weekly time step']
for i in range(0,len(WTP_weekly)):
    year=int(WTP_weekly['Time step'][i][:4])
    cond=WTP['Year']==year
    WTP_weekly.loc[i, 'WTP_HH'] = float(WTP['WTP Households (DKK/m3)'][cond].iloc[0])
    WTP_weekly.loc[i, 'WTP_Ind'] = float(WTP['WTP Industry (DKK/m3)'][cond].iloc[0])
    WTP_weekly.loc[i, 'WTP_PS'] = float(WTP['WTP Services (DKK/m3)'][cond].iloc[0])
    WTP_weekly.loc[i, 'WTP_Agri'] = float(WTP['WTP Agriculture (DKK/m3)'][cond].iloc[0])
    

# =============================================================================
# Linear reservoirs initial parameters
# =============================================================================

K_optim['Sinigwc'] = 1000   # 1000m3
K_optim['minbf'] = 10        # 1000 m3/week # doesn't work with min baseflow higher than 0 because some catchments are dry during summer...

K_optim['K'] = K_optim['K']
Kgwc=dict(zip(K_optim['Catchment'], K_optim['K']))  # K parameter weeks
if error_func != 'NSE' and Catch!='Test':
    Kgwc[2] = 10.85 # put a more realistic value for catch 2 (the value from the NSE optim)

Sinigwc = dict(zip(K_optim['Catchment'], K_optim['Sinigwc'])) # Storage intitial GW 1000 m3
minbf = dict(zip(K_optim['Catchment'], K_optim['minbf'])) # min Baseflow 1000 m3§/week 

Qbaseini = dict()  #initial BaseFlow 1000 m3/week
for c in ncatch:
    Qbaseini[c] = Sinigwc[c]/Kgwc[c]  

# =============================================================================
# Water Demand and WTP per WSA
# =============================================================================

D_HH = dict()
D_Ind = dict()
D_PS = dict()
D_Agri = dict()

WTP_HH = dict()
WTP_Ind = dict()
WTP_PS = dict()
WTP_Agri = dict()

for w in nwsa:
    for t in ntimes:
        
        D_HH[w,t] = float(WSA['Wateruse households (1000m3)'][WSA['WSAID']==w].iloc[0])/52.18  #avg wateruse of HH 1000m3 / year converted to 1000 m3 per week
        D_Ind[w,t] = float(WSA['Wateruse industries (1000m3)'][WSA['WSAID']==w].iloc[0])/52.18
        D_PS[w,t] = float(WSA['Wateruse services (1000m3)'][WSA['WSAID']==w].iloc[0])/52.18
        D_Agri[w,t] = float(WSA['Wateruse agriculture (1000m3)'][WSA['WSAID']==w].iloc[0])/52.18
        
        # WTP varying with time
        # WTP_HH[w,t] = WTP_weekly['WTP_HH'][t-1]
        # WTP_Ind[w,t] = WTP_weekly['WTP_Ind'][t-1]
        # WTP_PS[w,t] = WTP_weekly['WTP_PS'][t-1]
        # WTP_Agri[w,t] = WTP_weekly['WTP_Agri'][t-1]
        
        # current WTP for all timesteps
        # WTP_HH[w,t] = WTP_weekly['WTP_HH'].tolist()[-1]
        # WTP_Ind[w,t] = WTP_weekly['WTP_Ind'].tolist()[-1]
        # WTP_PS[w,t] = WTP_weekly['WTP_PS'].tolist()[-1]
        # WTP_Agri[w,t] = WTP_weekly['WTP_Agri'].tolist()[-1]
        
        # WTP constant, priorization: HH > PS > Ind > Agri
        WTP_HH[w,t] = WTP_weekly['WTP_HH'].tolist()[-1]
        WTP_Ind[w,t] = WTP_weekly['WTP_HH'].tolist()[-1]*1/3
        WTP_PS[w,t] = WTP_weekly['WTP_HH'].tolist()[-1]*1/2
        WTP_Agri[w,t] = WTP_weekly['WTP_Agri'].tolist()[-1]*1/4
        
        
# =============================================================================
# Inflow dictionary
# =============================================================================

I_inflow = dict()
for c in ncatch:
    for t in ntimes:
        I_inflow[c,t] = inflow[c-1,t-1]

# =============================================================================
# Max pumping capacity
# =============================================================================

maxpump = {}
for wf in nwf:
    # maxpump[wf] = WF.loc[WF['WFID'] == wf, 'AnlgTillad'].values[0]/52.18/1000 # weekly maxpump 1000m3
    maxpump[wf] = WF.loc[WF['WFID'] == wf, 'AnlgTillad'].values[0]/1000   # yearly maxpump 1000m3
    if Catch=='Test':
        maxpump[wf]=10*maxpump[wf] # to actually see something

# =============================================================================
# Storage capacity and initial storage of WaterWorks
# =============================================================================

# Change Tinghøj reservoir capacity : 250,000 m3 that is CPH demand for 2 days
WW.loc[WW['WWID'] == 1, 'Storage capacity (1000m3)'] = 250

maxstorage = {}
Storage_ini = {}
for ww in nww:
    maxstorage[ww] = WW.loc[WW['WWID'] == ww, 'Storage capacity (1000m3)'].values[0]
    Storage_ini[ww] = WW.loc[WW['WWID'] == ww, 'Storage initial (1000m3)'].values[0]
    
# =============================================================================
# Water exchange dictionary 
# =============================================================================

Water_Transfer = Water_Transfer.fillna(0) #replace nan with zeros in DataFrame
list_WW_transfer = Water_Transfer['AnlaegID'].tolist() # list of anlaeg involved in watertransfers

Water_Exchange_Capacity = {}  # capacities in 1000 m3
Water_Exchange_Links = {}   # connections

for ww1 in nww:
    for ww2 in nww:
        Water_Exchange_Capacity[ww1,ww2] = 0
        Water_Exchange_Links[ww1,ww2] = 0
        if ww1 in list_WW_transfer and ww2 in list_WW_transfer:
            Water_Exchange_Capacity[ww1,ww2] = float(Water_Transfer[Water_Transfer['AnlaegID']==ww1][str(ww2)])*7  # transfer capacity per 1000m3/day converted to 1000m3/week
            if Water_Exchange_Capacity[ww1,ww2] > 0:
                Water_Exchange_Links[ww1,ww2] = 1

if scenario == 2:
    kommuner = set(WW_municipality['KOMMUNENR2007'])    
    for k in kommuner:
        WW_in_K = WW_municipality[WW_municipality['KOMMUNENR2007']==k]['ANLAEGID'].tolist()
        WW_in_K = [x for x in WW_in_K if x in nww]    # remove WW not in nww
        # print('WW in', k, WW_municipality[WW_municipality['KOMMUNENR2007']==k]['KOMMUNENAVN'].to_list()[0], '=', len(WW_in_K))
        for i in WW_in_K:
            for j in WW_in_K:
                if Water_Exchange_Capacity[i,j]==0:
                    Water_Exchange_Capacity[i,j] = 2*7   # transfer caapcity of 2000 m3 per day (five time less than the big transfers)
                    Water_Exchange_Links[i,j] = 1
    print('Scenario ', scenario, ': water exchanges at the local level are added')


# =============================================================================
# WF to WW matrix to dict
# =============================================================================

WF_WW.set_index('WFID', inplace =True)
WF_WW_matrix = {(row, int(float(col))): int(WF_WW.at[row, col]) for row in WF_WW.index for col in WF_WW.columns}

# =============================================================================
# WW to WSA matrix to dict
# =============================================================================

WW_WSA.set_index('WWID', inplace =True)
WW_WSA_matrix = {(row, int(col)): int(WW_WSA.at[row, col]) for row in WW_WSA.index for col in WW_WSA.columns}

# =============================================================================
# Array with zeros for initialization of variables
# =============================================================================

zeros_ww_wsa_t = {}
for ww in nww:
    for wsa in nwsa:
        for t in ntimes:
            zeros_ww_wsa_t[ww,wsa,t] = 0
            
zeros_wf_ww_t = {}
for wf in nwf:
    for ww in nww:
        for t in ntimes:
            zeros_wf_ww_t[wf,ww,t] = 0            
            
# =============================================================================
# Create the year / week dictionnary
# =============================================================================

week_in_year = {}
for t in ntimes:
    week_in_year[t]=int(t//52.18)+1 

# =============================================================================
# Create a month / week dictionnary
# =============================================================================

week_in_month = {}
for t in ntimes:
    week_in_month[t] = int((t % 52.18)//(52.18/12)+1)

# =============================================================================
# Loss fraction wastewater (what goes to the sea)           
# =============================================================================

loss_fraction={}
for c in ncatch:
    loss_fraction[c] = 0.7

if Catch_Geo == 'Watersheds_Capital':
    loss_fraction[16] = 1   # Copenhagen
    loss_fraction[27] = 1   # Amager, taarnby
    
# =============================================================================
# Natural flow data (Q_natural) for comparison with the baseflow from the model
# =============================================================================

def linres(Sini, deltat, K, Inflow):
    nper = len(Inflow)
    Qout = np.zeros(nper+1)
    Qout[0] = Sini/K
    for i in range(nper):
        Qout[i+1] = Qout[i]*np.exp(-deltat/K) + Inflow[i]*(1-np.exp(-deltat/K))
    Qout = Qout[1:]
    return Qout

Q_natural = pd.DataFrame(index=ntimes)
for c in ncatch:
    K = K_optim['K'][c-1]
    Q_natural['Q_base_'+str(c)] = linres(1000,1,K,inflow[c-1])  # timestep 1 week for K in weeks

# minbf = dict(zip(ncatch, 0.75*Q_natural.median()))  # new minbf based on median natural baseflow
minbf = dict(zip(ncatch, 0.75*Q_natural.mean()))  # new minbf based on average natural baseflow

# =============================================================================
# Area dictionnary
# =============================================================================
    
area={}
for c in ncatch:
    area[c] = Catchments.loc[Catchments['Catch_ID']==c, 'Area (m2)'].values[0]


#%% Create Pyomo Model

# =============================================================================
# Create the model
# =============================================================================
print(time.strftime("%H:%M:%S") + ' Creating the model...')
model = ConcreteModel() # define the model

# =============================================================================
# Define the index 
# =============================================================================

model.ntimes = Set(initialize=ntimes) # define time index, set values to ntimes
model.nyear = Set(initialize=nyear)   # define year index
model.nwsa = Set(initialize=nwsa)     # define WSA index, set values to nwsa
model.nww = Set(initialize=nww)     # define WaterWorks index, set values to nww
model.nwf = Set(initialize=nwf)     # define Wellfields index, set values to nwf
model.ncatch = Set(initialize=ncatch) # define catchment index, set values to ncatch

# =============================================================================
# Declare decision variables - decision variable values will be provided by the optimizer
# =============================================================================

model.A_HH  = Var(model.nwsa, model.ntimes, within=NonNegativeReals) # Allocation to households, 1000 m3 per weekly time step
model.A_Ind  = Var(model.nwsa, model.ntimes, within=NonNegativeReals) # Allocation to Industry, 1000 m3 per weekly time step
model.A_PS  = Var(model.nwsa, model.ntimes, within=NonNegativeReals) # Allocation to Public services, 1000 m3 per weekly time step
model.A_Agri = Var(model.nwsa, model.ntimes, within=NonNegativeReals) # Allocation to Agriculture, 1000 m3 per weekly time step

model.Pump_WF = Var(model.nwf, model.ntimes, within=NonNegativeReals) # Sum of groundwater pumping for each wellfields 1000 m3 per weekly time step
model.Pump_catch = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # Sum of groundwater pumping for each catchment 1000 m3 per weekly time step
model.Pump_GW_to_BF = Var(model.ncatch, model.ntimes, within=NonNegativeReals)  # Pumping to the river to maintain a min BF 
model.Supply_WF_WW = Var(model.nwf,model.nww, model.ntimes, within=NonNegativeReals, initialize = zeros_wf_ww_t) # Supply from WF to WW 1000m3/week
model.Storage_WW = Var(model.nww, model.ntimes, within=NonNegativeReals) # Water storage for each waterworks 1000m3 per weekly time step 
model.Exchange  = Var(model.nww, model.nww, model.ntimes, within=NonNegativeReals) # Water transfer from 1 anlaeg to another, therefore 2 times nanlaeg, 1000m3 per weekly time step
model.Supply_WW_WSA = Var(model.nww, model.nwsa, model.ntimes, within=NonNegativeReals, initialize = zeros_ww_wsa_t) # Water Supply distributed by each Waterworks to all the WSA it serves

model.Q_base  = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # Base flow from GW catchment, 1000 m3 per weekly time step
model.Send   = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # One end storage per month and per reservoir. 1000 m3 per weekly time step

# =============================================================================
# Declare parameters
# =============================================================================

#model.endtime = Param(initialize = ntimes[-1]) # find end time step of the model
model.D_HH  = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = D_HH) # Set Houshold water demand to observed household water use, 1000 m3 per weekly time step
model.D_Ind = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = D_Ind) # Set Industry water demand to observed Industry water use, 1000 m3 per weekly time step
model.D_PS = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = D_PS) # Set Public services water demand to observed Energy supply water use, 1000 m3 per weekly time step
model.D_Agri = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = D_Agri) # Set Agriculture water demand to observed water supply water use, 1000 m3 per weekly time step

model.WTP_HH  = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = WTP_HH) # Set Willingness To Pay for the same use categories
model.WTP_Ind = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = WTP_Ind) # 
model.WTP_PS = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = WTP_PS) # 
model.WTP_Agri = Param(model.nwsa, model.ntimes,within=NonNegativeReals,initialize = WTP_Agri) # 

model.maxpump_WF = Param(model.nwf, within=NonNegativeReals,initialize = maxpump)  # Abstraction license 1000 m3/week
model.WF_WW = Param(model.nwf, model.nww, within=NonNegativeReals,initialize = WF_WW_matrix) # connections between WF and WW
model.Storage_WW_ini = Param(model.nww, within=NonNegativeReals,initialize = Storage_ini) # Initial storage in Waterworks 1000m3
model.maxstorage_WW = Param(model.nww, within=NonNegativeReals,initialize = maxstorage) # Max storage capacity 1000 m3
# model.exchange_links =Param(model.nww, model.nww, within=NonNegativeReals, initialize = Water_Exchange_Links) # Water exchange connections ( matrix with 1 and 0) : pb using it, we don't see SP !!!
model.maxexchange = Param(model.nww, model.nww, within=NonNegativeReals,initialize = Water_Exchange_Capacity)    # Water transfer capacity between waterworks in 1000m3/week
model.WW_WSA = Param(model.nww, model.nwsa, within=NonNegativeReals, initialize = WW_WSA_matrix) # connections between WW and WSA

model.pumping_cost = Param(within=NonNegativeReals, initialize = 1)   # pump cost in DKK/m3 or thousand DKK/1000m3
model.exchange_cost = Param(within=NonNegativeReals, initialize = 1)  # water exchange cost DKK/m3 per distance ?????8
model.loss_fraction_waste = Param(model.ncatch, within=NonNegativeReals, initialize = loss_fraction) #loss fraction of wastewater return flow to the river (what goes to the sea....)

model.Sinigwc = Param(model.ncatch, within=NonNegativeReals,initialize = Sinigwc) # Set initial GW storage for all groundwater catchments
model.Kgwc = Param(model.ncatch, within=NonNegativeReals,initialize = Kgwc) # Set time constant for all groundwater catchments
model.Qbaseini = Param(model.ncatch, within=NonNegativeReals,initialize = Qbaseini) # Set initial BaseFlow for all catchments
model.inflow = Param(model.ncatch, model.ntimes,within=Reals,initialize = I_inflow) # Set inflow to GW to for all catchments (from MIKE SHE model)
model.minbflow = Param(model.ncatch,within=Reals,initialize = minbf) # Set environmental constraint on flow for all catchments


#%% Set up the model

print(time.strftime("%H:%M:%S") + ' Defining the constraints...')

# =============================================================================
# Objective function
# =============================================================================

# Maximize the benefits and minimize the costs
def obj_rule(model):
    HH_ben = sum(model.WTP_HH[w,t]*model.A_HH[w,t] for w in model.nwsa for t in model.ntimes)
    Ind_ben = sum(model.WTP_Ind[w,t]*model.A_Ind[w,t]  for w in model.nwsa for t in model.ntimes)
    PS_ben = sum(model.WTP_PS[w,t]*model.A_PS[w,t]  for w in model.nwsa for t in model.ntimes)
    Agri_ben = sum(model.WTP_Agri[w,t]*model.A_Agri[w,t]  for w in model.nwsa for t in model.ntimes)
    Pump_cost = sum(model.pumping_cost*model.Pump_WF[wf,t]  for wf in model.nwf for t in model.ntimes) + sum(model.pumping_cost*model.Pump_GW_to_BF[c,t] for c in model.ncatch for t in model.ntimes)
    Exchange_cost = sum(model.exchange_cost*model.Exchange[ww1,ww2,t] for ww1 in model.nww for ww2 in model.nww for t in model.ntimes)
    return HH_ben + Ind_ben + PS_ben + Agri_ben - Pump_cost - Exchange_cost 
model.obj = Objective(rule=obj_rule, sense = maximize)

# =============================================================================
# Allocation constraints
# =============================================================================

# Household allocation does not exceed household demand. Active for every time step and catchment, thus two indices
def wd_HH_c(model, w, t):
    return model.A_HH[w,t] <= model.D_HH[w,t]
model.wd_HH = Constraint(model.nwsa, model.ntimes, rule=wd_HH_c)

# Industrial demand constraint per catchment. Active for every time step and catchment, thus two indices
def wd_Ind_c(model, w, t):
    return model.A_Ind[w, t] <= model.D_Ind[w,t]
model.wd_Ind = Constraint(model.nwsa, model.ntimes, rule=wd_Ind_c)

# Public services demand constraint per catchment. Active for every time step and catchment, thus two indices
def wd_PS_c(model, w, t):
    return model.A_PS[w,t] <= model.D_PS[w,t]
model.wd_PS = Constraint(model.nwsa, model.ntimes, rule=wd_PS_c)

# Agriculture demand constraint per catchment. Active for every time step and catchment, thus two indices
def wd_Agri_c(model, w, t):
    return model.A_Agri[w,t] <= model.D_Agri[w,t]
model.wd_Agri = Constraint(model.nwsa, model.ntimes, rule=wd_Agri_c)


# =============================================================================
# Pumping constraints 
# =============================================================================

# Pump_catch variable = Total pumping of the WellFields in one catchment
def pumping_catch_c(model,c,t):
    # get the list of WF in each catchment
    WF_in_catch = WF.drop_duplicates(subset={'WFID'})[WF.drop_duplicates(subset={'WFID'})['Catch_ID'] == c]['WFID'].values
    return  model.Pump_catch[c,t] == sum([model.Pump_WF[wf,t] for wf in WF_in_catch]) + model.Pump_GW_to_BF[c,t]
model.pumping_catch = Constraint(model.ncatch, model.ntimes, rule=pumping_catch_c)

# Total pumping for each Wellfields always below the abstraction license each week
# def pumping_WF_c(model, wf, t):
#     return model.Pump_WF[wf,t] <= model.maxpump_WF[wf]/52.18
# model.pumping_WF = Constraint(model.nwf, model.ntimes, rule=pumping_WF_c)

# Total pumping for each Wellfields always below the abstrction license for each year
def pumping_WF_c(model, wf, y):
    list_weeks = np.array([week[0] for week in week_in_year.items() if week[1] == y])  #get the list of the weeks in a given year
    return sum(model.Pump_WF[wf,t] for t in list_weeks) <= model.maxpump_WF[wf]
model.pumping_WF = Constraint(model.nwf, model.nyear, rule=pumping_WF_c)


# =============================================================================
# WaterBalance at the Wellfields, WaterWorks and WSA level
# =============================================================================

# At the WF level
def wb_WF_c(model,wf,t):
    return model.Pump_WF[wf,t] == sum(model.WF_WW[wf,ww]*model.Supply_WF_WW[wf,ww,t] for ww in model.nww)
model.wb_WF = Constraint(model.nwf, model.ntimes, rule=wb_WF_c)

# Set the value of Supply_WF_WW to zero when there is no connexion between WF and WW
def wb_max_WF_WW_c(model,wf,ww,t):
    return model.Supply_WF_WW[wf,ww,t] <= model.WF_WW[wf,ww]*10000   # multiply matrix by a high enough number so it's like an infinite transfer capacity in the pipes
# model.wb_max_WF_WW = Constraint(model.nwf, model.nww, model.ntimes, rule=wb_max_WF_WW_c)

# DeltaStorage = sum(Pumping)  +- Exchange  - Supply_WW_to_WSA
def wb_WW_c(model,ww,t):
    # get the list of the WF connected to the WW
    WF_in_WW = WF[WF['WWID'] == ww]['WFID'].values
    if t == 1:
        return model.Storage_WW[ww,t] - model.Storage_WW_ini[ww] == sum(model.WF_WW[wf,ww]*model.Supply_WF_WW[wf,ww,t] for wf in WF_in_WW) - sum(model.WW_WSA[ww,wsa]*model.Supply_WW_WSA[ww,wsa,t] for wsa in model.nwsa) + sum(model.Exchange[sender,ww,t] for sender in model.nww) - sum(model.Exchange[ww,receiver,t] for receiver in model.nww)
    else:
        return model.Storage_WW[ww,t] - model.Storage_WW[ww,t-1] == sum(model.WF_WW[wf,ww]*model.Supply_WF_WW[wf,ww,t] for wf in WF_in_WW) - sum(model.WW_WSA[ww,wsa]*model.Supply_WW_WSA[ww,wsa,t] for wsa in model.nwsa) + sum(model.Exchange[sender,ww,t] for sender in model.nww) - sum(model.Exchange[ww,receiver,t] for receiver in model.nww)
model.wb_WW = Constraint(model.nww, model.ntimes, rule=wb_WW_c)

#For each WaterWorks, Storage WW <= maxstorage WW
def wb_WW_Storage_c(model, ww, t):
    return model.Storage_WW[ww,t] <= model.maxstorage_WW[ww]
model.wb_WW_Storage = Constraint(model.nww, model.ntimes, rule=wb_WW_Storage_c)

# Max exchange capacity
def wb_WW_Exchange_c(model,ww1,ww2,t):
    return model.Exchange[ww1,ww2,t] <= model.maxexchange[ww1,ww2]
model.wb_WW_Exchange = Constraint(model.nww, model.nww, model.ntimes, rule=wb_WW_Exchange_c)
    
# Water allocation in each WSA, should be lower than the sum of the water allocated in the WSA
def wb_WSA_c(model, wsa, t):
    # get the list of WW delivering water to each WSA
    # WW_in_WSA = WW[WW['WSAID'] == wsa]['WWID'].values # try both with summing for ww in WW_in_WSA or ww in model.nww and compare time
    return model.A_HH[wsa,t] + model.A_Ind[wsa,t] + model.A_PS[wsa,t] + model.A_Agri[wsa,t] == sum(model.WW_WSA[ww,wsa]*model.Supply_WW_WSA[ww,wsa,t] for ww in model.nww)
model.wb_WSA = Constraint(model.nwsa, model.ntimes, rule=wb_WSA_c)

# Set the value of Supply_WW_WSA to zero when there is no connexion between WW and WSA
def wb_max_WW_WSA_c(model,ww,wsa,t):
    return model.Supply_WW_WSA[ww,wsa,t] <= model.WW_WSA[ww,wsa]*10000   # multiply matrix by a high enough number so it's like an infinite transfer capacity in the pipes
# model.wb_max_WW_WSA = Constraint(model.nww,model.nwsa, model.ntimes, rule=wb_max_WW_WSA_c)


# =============================================================================
# Linear reservoir constraints 
# =============================================================================


# Linear reservoirs base flow (from 1st order equation)
def lin_res_c(model,c,t):
    WSA_in_catch = WSA[WSA['Catch_ID'] == c]['WSAID'].values  # list all WSA in the catchment (for wastewater)
    if t == 1:
        return model.Q_base[c,t] == model.Qbaseini[c]*np.exp(-1/model.Kgwc[c]) + (model.inflow[c,t]-model.Pump_catch[c,t])*(1-np.exp(-1/model.Kgwc[c])) + (1-model.loss_fraction_waste[c])*sum([model.A_HH[wsa,t] + model.A_Ind[wsa,t] + model.A_PS[wsa,t] + model.A_Agri[wsa,t] for wsa in WSA_in_catch]) + model.Pump_GW_to_BF[c,t]
    else:
        return model.Q_base[c,t] == model.Q_base[c,t-1]*np.exp(-1/model.Kgwc[c]) + (model.inflow[c,t]-model.Pump_catch[c,t])*(1-np.exp(-1/model.Kgwc[c])) + (1-model.loss_fraction_waste[c])*sum([model.A_HH[wsa,t] + model.A_Ind[wsa,t] + model.A_PS[wsa,t] + model.A_Agri[wsa,t] for wsa in WSA_in_catch]) + model.Pump_GW_to_BF[c,t]
    
    # return model.Q_base[c,t] == Q_natural['Q_base_'+str(c)].mean()
    
model.lin_res = Constraint(model.ncatch, model.ntimes, rule=lin_res_c)  

# Linear reservoirs storage (from 1st order equation), not needed, only for plotting
def lin_res_stor_c(model,c,t):
    if t == 1:
        return model.Send[c,t] == model.Sinigwc[c]*np.exp(-1/model.Kgwc[c]) + (model.inflow[c,t]-model.Pump_catch[c,t])*(1-np.exp(-1/model.Kgwc[c]))*Kgwc[c]
    else:
        return model.Send[c,t] == model.Send[c,t-1]*np.exp(-1/model.Kgwc[c]) + (model.inflow[c,t]-model.Pump_catch[c,t])*(1-np.exp(-1/model.Kgwc[c]))*Kgwc[c]

    # return model.Send[c,t] == model.Sinigwc[c]

model.lin_res_stor = Constraint(model.ncatch, model.ntimes, rule=lin_res_stor_c)


# =============================================================================
# Environmental constraints
# =============================================================================

# min baseflow
def min_bf_c(model,c,y):
    list_weeks = [week[0] for week in week_in_year.items() if week[1] == y]  #get the list of the weeks in a given year
    return sum(model.Q_base[c,t] for t in list_weeks)/len(list_weeks) >= model.minbflow[c]   # yearly average flow above minBF
model.min_bf = Constraint(model.ncatch, model.nyear, rule=min_bf_c) 


# Max pumping of aquifer: from "Model and Ensemble Indicator-Guided Assessment of Robust,
#                              Exploitable Groundwater Resources for Denmark" table 1, indicator 2
def gw_ind_2_c(model,c,y):
    list_weeks = [week[0] for week in week_in_year.items() if week[1] == y]  #get the list of the weeks in a given year
    if ind_2 == True:
        return sum(model.Pump_catch[c,t] for t in list_weeks) <= sum(model.inflow[c,t] for t in list_weeks)/2
    else: # no constraint
        return sum(model.Pump_catch[c,t] for t in list_weeks) <= sum(model.inflow[c,t] for t in list_weeks)*2
model.gw_ind_2 = Constraint(model.ncatch, model.nyear, rule=gw_ind_2_c)


# =============================================================================
# Dual problem
# =============================================================================

# formulate dual problem to provide shadow prices
model.dual = Suffix(direction=Suffix.IMPORT) 

#%% Solve the model

print(time.strftime("%H:%M:%S") + ' Solving the model...')

# =============================================================================
# Create a solver
# =============================================================================

opt =SolverFactory(solvername,executable=solverpath_exe)
# opt = SolverFactory(solvername) # if it works without stating the path of the solver

# =============================================================================
# Solve
# =============================================================================

results = opt.solve(model)
end = time.time()

# =============================================================================
# Check status
# =============================================================================

print(time.strftime("%H:%M:%S") + ' Model solved! \n', 'Total computation time = ', round((end-start)/60,1), ' minutes')

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    # Do something when the solution in optimal and feasible
    print('Optimal and Feasible \n')
elif (results.solver.termination_condition == TerminationCondition.infeasible):
    # Do something when model in infeasible
    print('Infeasible')
else:
    # Something else is wrong
    print ('Solver Status: ',  results.solver.status)
    print ('Solver termination condition: ', results.solver.termination_condition)


#%% Output

# =============================================================================
# # Objective value
# =============================================================================

print("Total Benefit in optimal solution: ", round(value(model.obj)/len(model.ntimes)), " thousand DKK per week \n")


# =============================================================================
# Some results
# =============================================================================

A_HH_tot = sum(value(model.A_HH[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
A_Ind_tot = sum(value(model.A_Ind[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
A_PS_tot = sum(value(model.A_PS[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
A_Agri_tot = sum(value(model.A_Agri[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
A_tot = A_HH_tot + A_Ind_tot + A_PS_tot + A_Agri_tot

D_HH_tot = sum(value(model.D_HH[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
D_Ind_tot = sum(value(model.D_Ind[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
D_PS_tot = sum(value(model.D_PS[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
D_Agri_tot = sum(value(model.D_Agri[wsa,t]) for wsa in model.nwsa for t in model.ntimes)
D_tot = D_HH_tot + D_Ind_tot + D_PS_tot +D_Agri_tot

print('Allocation HH (%demand) : ',round(100*A_HH_tot/D_HH_tot,1), '%')
print('Allocation Ind (%demand) : ',round(100*A_Ind_tot/D_Ind_tot,1), '%')
print('Allocation PS (%demand) : ',round(100*A_PS_tot/D_PS_tot,1), '%')
print('Allocation total (%demand)',round(100*A_tot/D_tot,1), '%')
# print('Pumping (%maxpump) : ',round(100*sum(value(model.Pump_WF[wf,t]) for wf in model.nwf for t in model.ntimes)/sum(value(model.maxpump_WF[wf])/52.18 for wf in model.nwf for t in model.ntimes),1), '%')
# print('Pumping Catchments (1000m3) : ', round(sum(value(model.Pump_catch[c,t]) for c in model.ncatch for t in model.ntimes),1))
# print('Pumping GW to BF (1000m3)',round(sum(value(model.Pump_GW_to_BF[c,t]) for c in model.ncatch for t in model.ntimes),1))
# print('Total pumped at WF (1000m3) : ', round(sum(value(model.Pump_WF[wf,t]) for wf in model.nwf for t in model.ntimes),1))
# print('Total Supply WF to WW (1000m3) : ', round(sum(value(model.Supply_WF_WW[wf,ww,t]) for wf in model.nwf for ww in model.nww for t in model.ntimes),1))
# print('Total Supply WW to WSA (1000m3) : ', round(sum(value(model.Supply_WW_WSA[ww,wsa,t]) for ww in model.nww for wsa in model.nwsa for t in model.ntimes),1))
# print('Total allocated to users (1000m3) : ', round(A_HH_tot + A_Ind_tot + A_PS_tot + A_Agri_tot,1))
# print('Check max transfer WW to WSA (1000m3/week): ', max([value(model.Supply_WW_WSA[ww,wsa,t]) for ww in model.nww for wsa in model.nwsa for t in model.ntimes]))

A_HH_per_WSA = [100*sum(value(model.A_HH[wsa,t]) for t in model.ntimes)/(sum(value(model.D_HH[wsa,t]) for t in model.ntimes)+0.000000001) for wsa in model.nwsa]
A_Ind_per_WSA = [100*sum(value(model.A_Ind[wsa,t]) for t in model.ntimes)/sum(value(model.D_Ind[wsa,t]) for t in model.ntimes) for wsa in model.nwsa]
A_PS_per_WSA = [100*sum(value(model.A_PS[wsa,t]) for t in model.ntimes)/sum(value(model.D_PS[wsa,t]) for t in model.ntimes) for wsa in model.nwsa]
Allocation_per_WSA = pd.DataFrame(data={'HH':A_HH_per_WSA, 'Ind':A_Ind_per_WSA, 'PS':A_PS_per_WSA})


#%% Save optimal decisions

print(time.strftime("%H:%M:%S") + ' Saving the results...')

os.chdir(savepath)
outfile = r'Optimal_Decision_'+Catch_Geo+'.xlsx'

# =============================================================================
# Process decision variabes data
# =============================================================================

optimal_A_HH = np.zeros((len(model.ntimes),len(model.nwsa)))
optimal_A_Ind = np.zeros((len(model.ntimes),len(model.nwsa)))
optimal_A_PS = np.zeros((len(model.ntimes),len(model.nwsa)))
optimal_A_Agri = np.zeros((len(model.ntimes),len(model.nwsa)))

optimal_Pump_WF = np.zeros((len(model.ntimes),len(model.nwf)))
optimal_Pump_catch = np.zeros((len(model.ntimes),len(model.ncatch)))
optimal_Pump_GW_to_BF = np.zeros((len(model.ntimes),len(model.ncatch)))
optimal_Storage_WW = np.zeros((len(model.ntimes),len(model.nww)))
optimal_Q_base = np.zeros((len(model.ntimes),len(model.ncatch)))
optimal_Send = np.zeros((len(model.ntimes),len(model.ncatch)))

optimal_Exchange = np.zeros((len(model.ntimes),len(model.nww), len(model.nww)))
optimal_Supply_WW_WSA = np.zeros((len(model.ntimes),len(model.nww), len(model.nwsa)))

for t in model.ntimes:
    
    for i in range(1,len(model.nwsa)+1):
        optimal_A_HH[t-1,i-1] = value(model.A_HH[model.nwsa.at(i),t])
        optimal_A_Ind[t-1,i-1] = value(model.A_Ind[model.nwsa.at(i),t])
        optimal_A_PS[t-1,i-1] = value(model.A_PS[model.nwsa.at(i),t])
        optimal_A_Agri[t-1,i-1] = value(model.A_Agri[model.nwsa.at(i),t])

    for i in range(1,len(model.ncatch)+1):
        optimal_Pump_catch[t-1,i-1] = value(model.Pump_catch[model.ncatch.at(i),t])
        optimal_Pump_GW_to_BF[t-1,i-1] = value(model.Pump_GW_to_BF[model.ncatch.at(i),t])
        optimal_Q_base[t-1,i-1] = value(model.Q_base[model.ncatch.at(i),t])
        optimal_Send[t-1,i-1] = value(model.Send[model.ncatch.at(i),t])
        
    for i in range(1,len(model.nwf)+1):
        optimal_Pump_WF[t-1,i-1] = value(model.Pump_WF[model.nwf.at(i),t])
        
    for i in range(1, len(model.nww)+1):
        optimal_Storage_WW[t-1,i-1] = value(model.Storage_WW[model.nww.at(i),t])
        
        for j in range(1,len(model.nww)+1):
            optimal_Exchange[t-1,i-1,j-1] = value(model.Exchange[model.nww.at(i),model.nww.at(j),t])
            
        for j in range(1, len(model.nwsa)+1):
            optimal_Supply_WW_WSA[t-1,i-1,j-1] = value(model.Supply_WW_WSA[model.nww.at(i),model.nwsa.at(j),t])

# =============================================================================
# Convert to DataFrame and save
# =============================================================================

optimal_A_HH = pd.DataFrame(optimal_A_HH, index=model.ntimes, columns=['A_HH_'+str(w) for w in model.nwsa])
optimal_A_Ind = pd.DataFrame(optimal_A_Ind, index=model.ntimes, columns=['A_Ind_'+str(w) for w in model.nwsa])
optimal_A_PS = pd.DataFrame(optimal_A_PS, index=model.ntimes, columns=['A_PS_'+str(w) for w in model.nwsa])
optimal_A_Agri = pd.DataFrame(optimal_A_Agri, index=model.ntimes, columns=['A_Agri_'+str(w) for w in model.nwsa])

optimal_Storage_WW = pd.DataFrame(optimal_Storage_WW, index=model.ntimes, columns=['Storage_WW_'+str(w) for w in model.nww])
optimal_Pump_WF = pd.DataFrame(optimal_Pump_WF, index=model.ntimes, columns=['Pump_WF_'+str(w) for w in model.nwf])
optimal_Pump_catch = pd.DataFrame(optimal_Pump_catch, index=model.ntimes, columns=['Pump_catch_'+str(c) for c in model.ncatch])
optimal_Pump_GW_to_BF = pd.DataFrame(optimal_Pump_GW_to_BF, index=model.ntimes, columns=['Pump_GW_to_BF_'+str(c) for c in model.ncatch])
optimal_Q_base = pd.DataFrame(optimal_Q_base, index=model.ntimes, columns=['Q_base_'+str(c) for c in model.ncatch])
optimal_Send = pd.DataFrame(optimal_Send, index=model.ntimes, columns=['Send_'+str(c) for c in model.ncatch])

optimal_time = pd.DataFrame({'time':[t for t in model.ntimes]}, index=model.ntimes)

optimal_Decision = pd.concat([optimal_time, optimal_A_HH, optimal_A_Ind, optimal_A_PS, optimal_A_Agri, optimal_Storage_WW, optimal_Pump_WF, optimal_Pump_catch, optimal_Pump_GW_to_BF, optimal_Q_base, optimal_Send], axis=1) 
optimal_Decision.to_excel(outfile,sheet_name = 'Decision variables')

# =============================================================================
# Convert 3 dimensional decision variables in list of DataFrame
# =============================================================================


optimal_Exchange = [pd.DataFrame(optimal_Exchange[:,:,j-1], index=model.ntimes, columns=['Exchange_'+str(w)+'_to_'+str(model.nww.at(j)) for w in model.nww]) for j in range(1,len(model.nww)+1)]
optimal_Supply_WW_WSA = [pd.DataFrame(optimal_Supply_WW_WSA[:,:,j-1], index=model.ntimes, columns=['Supply_'+str(w)+'_to_'+str(model.nwsa.at(j)) for w in model.nww]) for j in range(1,len(model.nwsa)+1)]



#%% Save shadow prices

os.chdir(savepath)
outfile =     savepath + os.sep + r'Shadow_Prices_'+Catch_Geo+'.xlsx'

# =============================================================================
# Process Shadow prices data
# =============================================================================

SP_wd_HH = np.zeros((len(model.ntimes),len(model.nwsa)))
SP_wd_Ind = np.zeros((len(model.ntimes),len(model.nwsa)))
SP_wd_PS = np.zeros((len(model.ntimes),len(model.nwsa)))
SP_wd_Agri = np.zeros((len(model.ntimes),len(model.nwsa)))

SP_pumping_WF = np.zeros((len(model.nyear),len(model.nwf)))
SP_wb_WW_Storage = np.zeros((len(model.ntimes),len(model.nww)))
SP_wb_WW_Exchange = np.zeros((len(model.ntimes),len(model.nww),len(model.nww)))

# and Supply_WW_WSA ???

SP_lin_res = np.zeros((len(model.ntimes),len(model.ncatch)))
SP_min_bf =np.zeros((len(model.nyear),len(model.ncatch)))
SP_gw_ind_2 = np.zeros((len(model.nyear),len(model.ncatch)))


for t in model.ntimes:
    
    for i in range(1,len(model.nwsa)+1):      
        SP_wd_HH [t-1,i-1] = model.dual[model.wd_HH[model.nwsa.at(i),t]]
        SP_wd_Ind [t-1,i-1] = model.dual[model.wd_Ind[model.nwsa.at(i),t]]
        SP_wd_PS [t-1,i-1] = model.dual[model.wd_PS[model.nwsa.at(i),t]]
        SP_wd_Agri [t-1,i-1] = model.dual[model.wd_Agri[model.nwsa.at(i),t]]
        
    for i in range(1,len(model.ncatch)+1):
        SP_lin_res [t-1,i-1] = model.dual[model.lin_res[model.ncatch.at(i),t]]
        
        
    for i in range(1, len(model.nww)+1):
        SP_wb_WW_Storage[t-1,i-1] = model.dual[model.wb_WW_Storage[model.nww.at(i),t]]
        
        for j in range(1, len(model.nww)+1):
            SP_wb_WW_Exchange[t-1,i-1,j-1] = model.dual[model.wb_WW_Exchange[model.nww.at(i),model.nww.at(j),t]]

for y in model.nyear:
    for i in range(1,len(model.nwf)+1):
        SP_pumping_WF[y-1,i-1] = model.dual[model.pumping_WF[model.nwf.at(i),y]]

    for i in range(1,len(model.ncatch)+1):
        SP_min_bf [y-1,i-1] = model.dual[model.min_bf[model.ncatch.at(i),y]]
        SP_gw_ind_2 [y-1,i-1] = model.dual[model.gw_ind_2[model.ncatch.at(i),y]]


# =============================================================================
# Convert SP to dataframe and save
# =============================================================================

SP_wd_HH = pd.DataFrame(SP_wd_HH, index=model.ntimes, columns=['SP_wd_HH_'+str(w) for w in model.nwsa])
SP_wd_Ind = pd.DataFrame(SP_wd_Ind, index=model.ntimes, columns=['SP_wd_Ind_'+str(w) for w in model.nwsa])
SP_wd_PS = pd.DataFrame(SP_wd_PS, index=model.ntimes, columns=['SP_wd_PS_'+str(w) for w in model.nwsa])
SP_wd_Agri = pd.DataFrame(SP_wd_Agri, index=model.ntimes, columns=['SP_wd_Agri_'+str(w) for w in model.nwsa])

SP_wb_WW_Storage = pd.DataFrame(SP_wb_WW_Storage, index=model.ntimes, columns=['SP_wb_WW_Storage_'+str(w) for w in model.nww])
SP_pumping_WF = pd.DataFrame(SP_pumping_WF, index=model.nyear, columns=['SP_pumping_WF_'+str(w) for w in model.nwf])

SP_lin_res = pd.DataFrame(SP_lin_res, index=model.ntimes, columns=['SP_lin_res_'+str(c) for c in model.ncatch])
SP_min_bf = pd.DataFrame(SP_min_bf, index=model.nyear, columns=['SP_min_bf_'+str(c) for c in model.ncatch])
SP_gw_ind_2 = pd.DataFrame(SP_gw_ind_2, index=model.nyear, columns=['SP_gw_ind_2_'+str(c) for c in model.ncatch])

SP_time = pd.DataFrame({'time':[t for t in model.ntimes]}, index=model.ntimes)

SP = pd.concat([SP_time, SP_wd_HH, SP_wd_Ind, SP_wd_PS, SP_wd_Agri, SP_wb_WW_Storage, SP_pumping_WF, SP_lin_res, SP_min_bf, SP_gw_ind_2], axis=1) 
SP.to_excel(outfile,sheet_name = 'Shadow prices')

# =============================================================================
# Convert 3 dimensional SP in list of DataFrame
# =============================================================================

SP_wb_WW_Exchange = [pd.DataFrame(SP_wb_WW_Exchange[:,:,j-1], index=model.ntimes, columns=['Exchange_'+str(w)+'_to_'+str(model.nww.at(j)) for w in model.nww]) for j in range(1,len(model.nww)+1)]



#%% Indices

list_WSA = [wsa for wsa in model.nwsa]
list_WW = [ww for ww in model.nww]
list_WF = [wf for wf in model.nwf]
list_catch = [c for c in model.ncatch]
list_year = [y for y in model.nyear]

from datetime import date
TIME = [date.fromisoformat(WB_data[0]['Start time of weekly time step'][t-1]) for t in optimal_time['time']]

#%% SAVE DATA WITH PICKLE 


with open(savepath+ os.sep + 'Scenario_' + str(scenario) + '_' + str(weeks) + '_weeks'  + '_indices.pkl', 'wb') as file:
    pickle.dump([TIME,list_WSA,list_WW,list_WF,list_catch,list_year], file)
    
with open(savepath+ os.sep + 'Scenario_' + str(scenario) + '_' + str(weeks) + '_weeks'  + '_decision_variables.pkl', 'wb') as file:
    pickle.dump([optimal_Decision, optimal_time, optimal_A_HH, optimal_A_Ind, optimal_A_PS, optimal_A_Agri, optimal_Storage_WW, optimal_Pump_WF, optimal_Pump_catch, optimal_Pump_GW_to_BF, optimal_Q_base, optimal_Send, optimal_Exchange, optimal_Supply_WW_WSA], file)

with open(savepath+ os.sep + 'Scenario_' + str(scenario) + '_' + str(weeks) + '_weeks'  + '_shadow_prices.pkl', 'wb') as file:
    pickle.dump([SP, SP_time, SP_wd_HH, SP_wd_Ind, SP_wd_PS, SP_wd_Agri, SP_wb_WW_Storage, SP_pumping_WF, SP_lin_res, SP_min_bf, SP_gw_ind_2, SP_wb_WW_Exchange], file)
    
print(time.strftime("%H:%M:%S") + ' Results saved!')

