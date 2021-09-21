import numpy as np
import matplotlib.pyplot as plt


class MftwdfaAlg:


    # cumulative-sum profile of interpolated data
    def prof(self):
        self.profile = self.data_interp.cumsum(keep_attrs=True)
        return self.profile



    # weighted fit to profile
    # inputs: i = index of point to fit, s = number of points in window
    # outputs: yhat = fitted y-value for i-th point

    def wfit(self,i,s):


        ### WEIGHTS MATRIX

        # total number of (interpolated) data points
        n_points = self.divs["Qn"]

        # diagonal weights
        diag_vals = np.zeros(n_points)
        for j in range(n_points):
            diag_vals[j] = (1 - ((i-j)/s)**2)**2 if np.abs(i-j) <= s else 0

        # weights matrix
        W = np.diag(diag_vals,k=0)



        ### DATA

        # linreg matrix - first column is ones, second column is time values
        X = np.vstack( (np.ones(n_points), self.data_interp.time.values) ).T

        # cumulative-sum profile (runs function on obj and uses result)
        Y = self.profile.values



        ### LINREG

        # linear regression coefficients
        a = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)

        # original and fitted values
        xval = self.profile[i].time.values
        yval = self.profile[i].values
        yhat = a[0] + a[1] * xval


        return yhat



    # sum variances up and down the profile
    def varsum(self):
        print("varsum")


    # fluctuation function
    def fluct(self):
        print("fluct")


    # full MFTWDFA algorithm
    def mftwdfa(self,s):

        yhats = np.zeros(self.divs["Qn"])
        for i in range(self.divs["Qn"]):
            yhats[i] = self.wfit(i,s)




        print("mftwdfa")

        plt.scatter(self.profile.time.values, self.profile.values)
        plt.plot(self.profile.time.values, yhats)
        plt.show()
