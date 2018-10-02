from __future__ import division

# from matplotlib.pyplot import step, xlim, ylim, show
# import matplotlib.pyplot as plt
import datetime
import pytz

from datetime import timedelta
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# xbos clients
from xbos import get_client
from xbos.services.hod import HodClient
from xbos.services.mdal import *

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

from house import IEC # Prediction model. 
import pdb
import IPython


now = datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(
    tz=pytz.timezone("America/Los_Angeles"))
print now
#now = datetime.strptime("2018-08-26 12:00:00",'%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.timezone("UTC")).astimezone(
 #   tz=pytz.timezone("America/Los_Angeles"))

end = now.strftime('%Y-%m-%d %H:%M:%S %Z')
start = (now - timedelta(days = 30)).strftime('%Y-%m-%d %H:%M:%S %Z') 
WINDOW = '1min' ##TODO: change to 5 

# data clients. 

# To get the client we usually need a client for BOSSWAVE our decentralized operating system.
# Easiest way to get it is by using get_client() which you import from xbos. Other ways include entity files. 
# https://github.com/SoftwareDefinedBuildings/XBOS. 
# To use xbos make sure to get an entity file from Thanos and to get a executable file which 
# connects you to the system. Also, make sure to set the entity file in your bash_profile with
# export BW2_DEFAULT_ENTITY=path to .ent file

# The MDAL client gets the data from our database. The query to get the data is illustrated by,
# buidling_meteres_query_mdal and lighting_meter_query_mdal.
# Documentation: https://docs.xbos.io/mdal.html#using and https://github.com/gtfierro/mdal <- better
mdal = MDALClient("xbos/mdal", client=get_client())
# HODClient gets the uuid for data. This uses brick which is a language built on SPARQL.
# Can be trick to use.
# To try your own queries go to: corbusier.cs.berkeley.edu:47808. And try the queries we set up below.
# Documentation: for brick: brickschema.org/structure/
# If you need queries, it's best to ask either Thanos or Daniel. 
hod = HodClient("xbos/hod")

# brick model -- resources. HOD has the metadata and the context. MDAL uses 

# temporal parameters

#
#
#

SITE = "ciee" # TODO: get a different site and zone. Go on Mr. Plotter, choose a different building, expand a tree and pick a zone

print "Site: " + SITE  

# Brick queries
building_meters_query = """SELECT ?meter ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Building_Electric_Meter .
    ?meter bf:uuid ?meter_uuid .
};"""
thermostat_state_query = """SELECT ?tstat ?status_uuid FROM %s WHERE {
    ?tstat rdf:type brick:Thermostat_Status .
    ?tstat bf:uuid ?status_uuid .
};"""
# lighting_state_query = """SELECT ?lighting ?state_uuid FROM %s WHERE {
#     ?light rdf:type brick:Lighting_State .
#     ?light bf:uuid ?state_uuid
# };"""
# lighting_meter_query = """SELECT ?lighting ?meter_uuid FROM %s WHERE {
#     ?meter rdf:type brick:Electric_Meter .
#     ?lighting rdf:type brick:Lighting_System .
#     ?lighting bf:hasPoint ?meter .
#     ?meter bf:uuid ?meter_uuid
# };"""

building_meters_query_mdal = {
    "Composition": ["meter", "tstat_state"],  # defined under "Variables"
    "Selectors": [MEAN, MAX],
    "Variables": [
        {
            "Name": "meter",
            "Definition": building_meters_query % SITE, # NOTE: Mdal queries the uuids by itself. 
            #it is better practice for now to do that manually by calling hod.do_query(your_query)
            "Units": "kW"
         },
        {
            "Name": "tstat_state",
            "Definition": thermostat_state_query % SITE,
        }
    ],
    "Time": {
        "T0": start, "T1": end,
        "WindowSize": WINDOW,
        "Aligned": True,
    }
}

resp = mdal.do_query(building_meters_query_mdal)

#IPython.embed()

df = resp['df']

demand = "4d6e251a-48e1-3bc0-907d-7d5440c34bb9" #--> for ciee
#demand = "b5ccea9a-7dfa-388c-9060-3adb55001f6e" # --> for harward_8


# lighting_meter_query_mdal = {
#     "Composition": ["lighting"],
#     "Selectors": [MEAN],
#     "Variables": [
#         {
#             "Name": "lighting",
#             "Definition": lighting_meter_query % SITE,
#             "Units": "kW"
#         },
#     ],
#     "Time": {
#         "T0": start, "T1": end,
#         "WindowSize": WINDOW,
#         "Aligned": True,
#     }
# }
# queries the data from the database with mdal
# resp = mdal.do_query(lighting_meter_query_mdal, timeout=120)
# lighting_df = resp['df']


def feed_forward_rmse(algorithm = "GP PerMatern52", future_window = 12 * 60, num_cuts = 12):
    all_indexes = map(int, range(0, future_window, int(future_window/num_cuts)))
    RMSE = np.array([])

    for train_index in all_indexes:
        X_train = train_index
        X_test = X_train + int(future_window / num_cuts)
        if future_window == X_test:
            y_test = meterdata[["House Consumption"]][ -future_window + X_train: ]
        else:    
            y_test = meterdata[["House Consumption"]][ -future_window + X_train : -future_window + X_test]
        
        # print(X_train, X_test, y_test.shape)

        model = IEC(meterdata[:(yesterday + timedelta(hours = train_index/60))].fillna(value=0), prediction_window = X_test)

        prediction_temp = model.predict([algorithm]).fillna(value = 0)

        A = prediction_temp[[algorithm]][X_train:X_test].values - y_test[["House Consumption"]].values
        A = A[~np.isnan(A)]
        RMSE = np.append(RMSE, np.sqrt(np.mean(A)**2))

    print(algorithm, np.mean(RMSE))
    return np.mean(RMSE)

def predictive_horizon_rmse(algorithm = "Baseline Finder", future_window = 12 * 60, num_cuts = 6):
    all_indexes = map(int, range(0, future_window, int(future_window/num_cuts)))
    RMSE = np.array([])
    X_train = 0
    
    yesterday = now - timedelta(hours = 12)
 
    for train_index in all_indexes:
        
        #X_train = train_index # --> ? 
        X_test = train_index + int(future_window / num_cuts)
        if future_window == X_test:
            y_test = meterdata[["House Consumption"]][ -future_window + X_train: ]
        else:    
            y_test = meterdata[["House Consumption"]][ -future_window + X_train : -future_window + X_test]
        
        yesterday = now - timedelta(hours = 12)+timedelta(minutes=X_train)


        # print(X_train, X_test, y_test.shape)

        # exclude the NAs or interpolate 

        model_RMSE = IEC(meterdata[:yesterday].fillna(value=0), prediction_window = X_test)
        prediction_temp = model_RMSE.predict([algorithm]).fillna(value = 0)

        A = prediction_temp[[algorithm]][X_train:X_test].values - y_test[["House Consumption"]].values
        A = A[~np.isnan(A)]
        
        RMSE = np.append(RMSE, np.sqrt(np.mean(A**2)))


    final = pd.DataFrame(all_indexes).rename(columns = {0:"index"})
    final["RMSE"] = RMSE

    return final 

# We are using a similarity based approach to predict. This means, that we build a similarity 
# measure to see how similar past days were to the day we are currently experiencing and taking
# a sort of weighted average. This is similar to k-nearest-neighbors. 
# Other approaches can also be tried, like finding different features by which to measure similarity
# or some other ML technique all together (e.g. neural nets or some regression)

# TODO Currently the big issue we are facing is that we are trying to subtract
# the variable consumption from the overall building consumption. This means that for our case
# we want to subtract the HVAC (heating) consumption from the building consumption, since that is 
# something we control and it doesn't make sense to learn it.
# Now, the issue is finding out what the exact consumption for heating and cooling is which Marco is doing and should
# be followed up with.

# TODO find out the right values. Marco is working on this
heating_consume = .3  # in kW
cooling_consume = 5.  # kW
meter = df.columns[0]
all_but_meter = df.columns[1:]

# amount to subtract for heating, cooling
# TODO some values become negative after the following operation, which should not happen. Marco is checking the data.
h = (df[all_but_meter] == 1).apply(sum, axis=1) * heating_consume
c = (df[all_but_meter] == 2).apply(sum, axis=1) * cooling_consume

meterdata = df[meter] - h - c

# NOTE: the following is some data manipulation and setting up the right time zone.
# Followed by, predicting the data with the IEC model. 

# unit conversion
meterdata = meterdata / float(1000 * 60)
# print meterdata
#print lighting_df.describe()
#IPython.embed()
meterdata = pd.DataFrame.from_records({'House Consumption': meterdata})
#print meterdata.describe()
#print meterdata['House Consumption']

meterdata = meterdata.tz_convert(pytz.timezone("America/Los_Angeles"))  # /??????????????????????????????????
meterdata["House Consumption"]= meterdata["House Consumption"].fillna(value = 0)
yesterday = now - timedelta(hours = 12) # ? 

# Prediction happening here and should be looked at.
#print(meterdata[:yesterday].fillna(value=0))

future_window = 12 * 60
model = IEC(meterdata[:yesterday].fillna(value = 0), prediction_window = future_window)
#algo_keys = ["GP PerExp", "GP PerMatern32", "GP PerMatern52", "ARIMA", "Baseline Finder"]#, "b1", "b2", "b3", "b4", "b5", "b6", "b7", "best4", "best12", "best24"]
# algo_keys = model.algorithms.keys()

algo_keys = ["Baseline Finder","residGP1", "residRNN_nolag"]

min_rmse = 1000000
best_algo_name = ""

print(meterdata)

future_window1 = 120 * 60
index = np.arange(future_window1)
plt.title("rawdata")
plt.plot(index, meterdata[["House Consumption"]][-future_window1:], label="Ground Truth")
plt.xlabel('Predictive horizon (Minutes)') # change this to hours 
plt.ylabel(r'KWh')
plt.show()

for algo_name in algo_keys:
    print(algo_name)
    # if algo_name[0] is not "b":
    #     continue
    # if "best" in algo_name:
    #     new_rmse = rmse(algo_name, future_window, int(algo_name.replace("best","")))
    # else:
    #     new_rmse = rmse(algo_name, future_window, 2)
    # if new_rmse < min_rmse:
    #     min_rmse = new_rmse
    #     best_algo_name = algo_name

    prediction = model.predict([algo_name])

 #   pred_RMSE = feed_forward_rmse(algorithm = algo_name, future_window= 12*60, num_cuts = 12)
    
    predictive_horizon = False

    if predictive_horizon:

        pred_RMSE = predictive_horizon_rmse(algorithm = algo_name, future_window= 12*60, num_cuts = 6)## make num_cuts more
        plt.plot(pred_RMSE[["index"]], pred_RMSE[["RMSE"]], label = algo_name)
        plt.xlabel('Predictive horizon (Minutes)')
        plt.ylabel(r'KWh')
        plt.legend(loc = 'upper left')
        
    
    else: 

        index = np.arange(future_window)
        plt.title(algo_name)
        plt.plot(index, prediction[[algo_name]], label="Energy Prediction")
        plt.plot(index, meterdata[["House Consumption"]][-future_window:], label="Ground Truth")
        plt.xlabel('Predictive horizon (Minutes)') # change this to hours 
        plt.ylabel(r'KWh')
        plt.legend()
        plt.savefig(algo_name + str(now)+'.png', bbox_inches='tight')
        plt.show()

plt.savefig("pred_horizon"+ algo_name +str(now)+ '.png', bbox_inches='tight')    
plt.show()

    


print("Minimum RMSE: " + best_algo_name)


####### notes from 8/31


# - ML algorithm without interpolation will be equivalent to one that does "dumber" work 
# - one extra vector to indicate which values are interpolating 
# - two questions -- why does the indicator functions worsen the model?

# ------- GP gets more nonconvex with more parameters
# ------- adding with GP, increase short term regression accuracy 
# ------- additional benefit of GP is stochasticity  

# for the consumption model -- trying to smooth the data, what if you don't have the last 
# 

# softmax rep used at output nodes of nn can model 








