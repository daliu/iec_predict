### this file is to generate the set of residuals for the GP to use 

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




now1 = datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(
	    tz=pytz.timezone("America/Los_Angeles"))

ending_times = [(now1-timedelta(hours = 24*i)) for i in range(100)]

master_resid_array = pd.DataFrame([[0,0]],columns = ["time", "resids"])

for now in ending_times:

	end = now.strftime('%Y-%m-%d %H:%M:%S %Z')
	start = (now - timedelta(days = 40)).strftime('%Y-%m-%d %H:%M:%S %Z') 
	WINDOW = '1min'

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

	# temporal parameters
	SITE = "ciee"

	# Brick queries
	building_meters_query = """SELECT ?meter ?meter_uuid FROM %s WHERE {
	    ?meter rdf:type brick:Building_Electric_Meter .
	    ?meter bf:uuid ?meter_uuid .
	};"""
	thermostat_state_query = """SELECT ?tstat ?status_uuid FROM %s WHERE {
	    ?tstat rdf:type brick:Thermostat_Status .
	    ?tstat bf:uuid ?status_uuid .
	};"""
	lighting_state_query = """SELECT ?lighting ?state_uuid FROM %s WHERE {
	    ?light rdf:type brick:Lighting_State .
	    ?light bf:uuid ?state_uuid
	};"""
	lighting_meter_query = """SELECT ?lighting ?meter_uuid FROM %s WHERE {
	    ?meter rdf:type brick:Electric_Meter .
	    ?lighting rdf:type brick:Lighting_System .
	    ?lighting bf:hasPoint ?meter .
	    ?meter bf:uuid ?meter_uuid
	};"""

	building_meters_query_mdal = {
	    "Composition": ["meter", "tstat_state"],  # defined under "Variables"
	    "Selectors": [MEAN, MAX],
	    "Variables": [
	        {
	            "Name": "meter",
	            "Definition": building_meters_query % SITE, # NOTE: Mdal queries the uuids by itself. it is better practice for now to do that manually by calling hod.do_query(your_query)
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
	df = resp['df']

	demand = "4d6e251a-48e1-3bc0-907d-7d5440c34bb9"


	lighting_meter_query_mdal = {
	    "Composition": ["lighting"],
	    "Selectors": [MEAN],
	    "Variables": [
	        {
	            "Name": "lighting",
	            "Definition": lighting_meter_query % SITE,
	            "Units": "kW"
	        },
	    ],
	    "Time": {
	        "T0": start, "T1": end,
	        "WindowSize": WINDOW,
	        "Aligned": True,
	    }
	}
	# queries the data from the database with mdal
	resp = mdal.do_query(lighting_meter_query_mdal, timeout=120)
	lighting_df = resp['df']

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
	meterdata = meterdata / (1000 * 60)
	# print meterdata
	#print lighting_df.describe()
	meterdata = pd.DataFrame.from_records({'House Consumption': meterdata})
	#print meterdata.describe()
	#print meterdata['House Consumption']

	meterdata = meterdata.tz_convert(pytz.timezone("America/Los_Angeles"))  # /??????????????????????????????????
	yesterday = now - timedelta(hours = 12) # ? 

	future_window = 12 * 60
	model = IEC(meterdata[:yesterday].fillna(value = 0), prediction_window = future_window)
	algo_name = ["Baseline Finder"]
	prediction = model.predict(algo_name)

	time_index = pd.DataFrame(meterdata.index[-future_window:]).rename(columns = {0:"time"})

	a = (meterdata[["House Consumption"]][-future_window:].fillna(value=0).values - \
		prediction[algo_name].fillna(value= 0).values)
	time_index["resids"] = a

	master_resid_array = master_resid_array.append(time_index)


pdb.set_trace()
master_resid_array.drop([0]).to_csv("resids.csv")












