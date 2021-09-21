from DataSet import DataSet

path = "C:/Users/ndbke/Dropbox/_NDBK/Research/epica_data/edc3/edc3-2008_co2_DATA-series3-composite.txt"
ds = DataSet(path)
ds.model_setup(periods=10,points=50)
ds.mftwdfa(s=10)
