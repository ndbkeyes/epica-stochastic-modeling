# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:56:55 2021

@author: ndbke
"""

import numpy as np
import pandas as pd
import plotly as pio
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import Akima1DInterpolator as akima


class DataSet:

    

    # constructor
    def __init__(self, name, path):
        
        
        ### READ & CLEAN DATA
        ts = pd.read_csv(path,sep='\s+')                                        # read CSV into pandas dataframe
        ts = ts.rename(columns={ts.columns[0]: "time", ts.columns[1]: "data"})  # rename columns to generic X, Y
        ts["time"] *= -1                                                        # make time axis values negative
        ts = ts.groupby(["time"],as_index=False).mean()                         # average Y-values with duplicate X-values
        ts = ts.set_index("time",drop=True)                                     # make X into dimension not variable 
        ts = ts.to_xarray()                                                     # convert pandas dataframe to xarray
        
        ### SET ATTRIBUTES
        self.data_ts = ts
        self.nametag = name

        


    # set up for modeling
    def model_setup(self,Pn,Pq):
        
        
        ### SETTINGS DICT

        sd = {}
        sd["t_start"] = -798000                                 # start time of data range
        sd["t_stop"] = 0                                        # stop time of data range
        sd["L"] = sd["t_stop"] - sd["t_start"]                  # time range (stop - start)
        sd["Pn"] = Pn                                           # number of periods
        sd["Pq"] = Pq                                           # number of points per periods
        sd["Py"] = sd["L"] / sd["Pn"]                           # length of period in years
        sd["Qn"] = sd["Pq"] * sd["Pn"]                          # number of points total
        sd["Qy"] = sd["L"] / sd["Qn"]                           # time between points in years
        
        # timestamps of all period starts
        sd["P_times"] = np.linspace(sd["t_start"], sd["t_stop"], sd["Pn"], endpoint=False)
        # timestamps of all points - to interpolate to!
        sd["Q_times"] = np.linspace(sd["t_start"], sd["t_stop"], sd["Qn"], endpoint=False)
        
        self.divs = sd


        print("XYZ")
        

        
        ### INTERPOLATE
        f = akima( self.data_ts.time.values, self.data_ts.data.values )             # interpolator function                                          
        ts_i = pd.DataFrame(data={"time": sd["Q_times"], "data": f(sd["Q_times"])}) # dataframe with interpolated time & data
        ts_i = ts_i.set_index("time",drop=True)                                     # index with time
        self.data_interp = ts_i.to_xarray()                                         # convert pandas to xarray
        
        # plot results
        self.data_ts.plot.scatter(x="time",y="data")
        self.data_interp.plot.scatter(x="time",y="data")

        

        ### DATA MATRIX
        
        # reshape time & data
        mat_shape = (sd["Pn"],sd["Pq"])
        time_reshape = np.reshape(self.data_interp.time.values, mat_shape)
        data_reshape = np.reshape(self.data_interp.data.values, mat_shape)
   
        # new xarray for matrixed data & times, with coordinates (p,q)
        self.data_matrix = xr.DataArray(
            data=data_reshape,
            coords=dict(
                p=range(sd["Pn"]),
                q=range(sd["Pq"]),
                time=( ["p","q"], time_reshape )
            ),
            dims=["p","q"]
        )
        
        
        
        
path = "C:/Users/ndbke/Dropbox/_NDBK/Research/epica_data/edc3/edc3-2008_co2_DATA-series3-composite.txt"
ds = DataSet("co2",path)
ds.model_setup(20,10)