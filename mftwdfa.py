import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.interpolate import Akima1DInterpolator as akima
from scipy.linalg import solve, inv
import warnings

warnings.filterwarnings('ignore', '.*ill-conditioned.*',)



class MftwdfaAlg:



    ##### interpolate data_ts values to t_new times
    # t_new = array of times to interpolate to
    def interp(self, t_new):

        f = akima( self.data_ts.time, self.data_ts )                    # interpolator function
        ts_i = pd.DataFrame(data={"time": t_new, "data": f(t_new)})     # pandas dataframe of interpolated time & data
        ts_i = ts_i.set_index("time",drop=True)                         # index with time
        ts_i = ts_i.to_xarray()                                         # convert pandas to xarray Dataset
        ts_i = ts_i["data"] - ts_i["data"].mean()                       # save DataArray with mean set to 0
        self.data_interp = ts_i



    ##### cumulative-sum profile of interpolated data
    def profile(self):
        self.data_profile = self.data_interp.cumsum(keep_attrs=True)
        return self.data_profile



    ##### weighted fit to profile
    # i = index of point to fit, s = size of window (in # of points)
    def wfit(self,i,s):


        ### WEIGHTS MATRIX
        # total number of (interpolated) data points
        n_points = len(self.data_profile)
        # diagonal weights
        diag_vals = [(1 - ((i-j)/s)**2)**2 if np.abs(i-j) <= s else 0 for j in range(n_points)]
        # weights matrix
        W = np.diag(diag_vals,k=0)

        ### DATA
        # linreg matrix - first column is ones, second column is time values
        X = np.vstack( (np.ones(n_points), self.data_profile.time.values) ).T
        Y = self.data_profile.values

        ### LINREG
        # linear regression coefficients
        A = X.T.dot(W).dot(X)
        b = X.T.dot(W).dot(Y)
        # coeffs = inv(A).dot(b)
        coeffs = solve(A,b)
        # get fitted y-value
        xval = self.data_profile[i].time.values
        yhat = coeffs[0] + coeffs[1] * xval
        return yhat



    ##### sum variances up and down the profile
    # nu = index of subdivision, s = size of subdivision (in # of points)
    def varsum(self,nu,s):

        # number of windows of size s, within N points total
        N = len(self.data_profile)
        Ns = int(np.floor(N/s))

        # select indices depending on direction up/down profile
        i1 = nu*s if nu < Ns else N - (nu+1-Ns)*s
        i2 = i1 + s

        # get actual and fitted y values (of profile) within nu-th window
        y = self.data_profile.values[i1:i2]
        yhat = [self.wfit(i,s) for i in range(i1,i2)]

        # sum up to get variance
        V = 1/s * np.sum( (y - yhat)**2 )
        return V



    ##### fluctuation function
    # s = timescale (in # of points), q = statistical moment to compute fluct func for
    def fluct(self,s,q):

        print(s)
        Ns = int(np.floor( len(self.data_profile) / s ))
        Fqs = ( 1/(2*Ns) * np.sum([self.varsum(nu,s)**(q/2) for nu in range(2*Ns)]) ) ** (1/q)
        return Fqs



    ##### write MFTWDFA fluct func to text file
    # flucts = pd DataFrame of fluctuation function (in log form)
    def write_ff(self,flucts):
        flucts.to_csv("../mftwdfa_ff.txt")



    ##### read MFTWDFA fluct func from text file
    def read_ff(self):
        fluct = pd.read_csv("../mftwdfa_ff.txt")
        fluct = fluct.set_index("log_s",drop=True)
        print(fluct)
        return fluct



    ###### full MFTWDFA algorithm
    # points = number of points to interpolate to / resolution or whatever
    def mftwdfa(self,points):

        # interpolation & profile building
        t_min = np.amin(self.data_ts.time.values)
        t_max = np.amax(self.data_ts.time.values)
        t_new = np.linspace(t_min,t_max,points)
        self.interp(t_new)
        self.profile()

        # fluct function over range of timescales
        s_arr = range( 2, int(np.floor(len(self.data_profile)/2)) )
        ff = [self.fluct(s,q=2) for s in s_arr]

        # save as DataFrame & write to text file
        flucts = pd.DataFrame( data={"log_fluct": np.log10(ff)}, index=np.log10(s_arr*(t_max-t_min)/points) )
        flucts.index.name = "log_s"
        self.write_ff(flucts)

        # plot log-log fluct plot
        flucts.plot()
        plt.show()



    ##### analyze slopes of fluctuation function
    def slope(self):
        # TODO
        print("slope analysis")
