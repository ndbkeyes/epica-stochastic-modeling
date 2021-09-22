import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from mftwdfa import MftwdfaAlg
from oumodel import OUModel




class DataSet(MftwdfaAlg,OUModel):



    # constructor
    def __init__(self, path):


        ### READ & CLEAN DATA
        ts = pd.read_csv(path,sep='\s+')                                        # read CSV into pandas dataframe
        ts = ts.rename(columns={ts.columns[0]: "time", ts.columns[1]: "data"})  # rename columns to generic time, data
        ts["time"] *= -1                                                        # make time axis values negative
        ts = ts.groupby(["time"],as_index=False).mean()                         # average data values with duplicate times
        ts = ts.set_index("time",drop=True)                                     # make time into dimension not variable
        ts = ts.to_xarray()                                                     # convert pandas dataframe to xarray

        ### SET AS ATTRIBUTE
        self.data_ts = ts["data"]




    # set up for OU modeling
    def model_setup(self,periods,points):



        ### SETTINGS DICT

        sd = {}
        sd["t_start"] = -798000                                 # start time of data range
        sd["t_stop"] = 0                                        # stop time of data range
        sd["L"] = sd["t_stop"] - sd["t_start"]                  # time range (stop - start)
        sd["Pn"] = periods                                      # number of periods
        sd["Pq"] = points                                       # number of points per period
        sd["Py"] = sd["L"] / sd["Pn"]                           # length of period in years
        sd["Qn"] = sd["Pq"] * sd["Pn"]                          # number of points total
        sd["Qy"] = sd["L"] / sd["Qn"]                           # time between points in years

        # timestamps of all period starts
        sd["P_times"] = np.linspace(sd["t_start"], sd["t_stop"], sd["Pn"], endpoint=False)
        # timestamps of all points - to interpolate to!
        sd["Q_times"] = np.linspace(sd["t_start"], sd["t_stop"], sd["Qn"], endpoint=False)

        self.divs = sd



        ### DATA MATRIX

        # interpolate to timestamps of all points
        self.interp(times=sd["Q_times"])

        # reshape time & data
        mat_shape = (sd["Pn"],sd["Pq"])
        time_reshape = np.reshape(self.data_interp.time.values, mat_shape)
        data_reshape = np.reshape(self.data_interp.values, mat_shape)

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



        # PLOTTING

        self.data_ts.plot()
        self.data_interp.plot()
        plt.title("EPICA ice core data - CO2")
        plt.legend(["original","interpolated"])
        plt.show()
