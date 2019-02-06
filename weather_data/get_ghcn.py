import pandas as pd
import numpy as np
from ftplib import FTP
from calendar import isleap

# cd Desktop/Projects/Data/GHCN/
# include FTP to get ghcnd-stations.txt?
# download from ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/
# load fixed width file 
colspecs = ((0,12),(12,21),(21,31),(31,38),
	    (38,41),(41,72),(72,80),(80,86))
meta = pd.read_fwf('ghcnd-stations.txt',
		    colspecs=colspecs,header=None,index_col=False,
		    names=['stationID','lat','lon','elev',
			   'state','name','network','val'])
		   
ushcn = meta[meta['network'].str.match('HCN',na=False)]

states = ['IL','IN','IA','KS','KY','MI','MN','MO','NE','OH','SD','WI']

stations = ushcn[ushcn.state.isin(states)].stationID.values.tolist()

# save location information for later
loc_var = ['lat','lon','elev']
tmp = ushcn.loc[ushcn.stationID.isin(stations),loc_var]
locs = pd.DataFrame(data=tmp.values,index=stations,columns=loc_var).T

stations = [s+'.dly' for s in stations]

ftp = FTP('ftp.ncdc.noaa.gov')
# email address should work for password
ftp.login(passwd='XXX@YYY')
ftp.cwd('/pub/data/ghcn/daily/hcn')
# make sure you're in the right directory
for stat in stations:
  ftp.retrbinary('RETR '+stat, open(stat,'wb').write)
  if int(stat[3:11]) % 5 == 0: print stat
ftp.close()

# extract variables with loop, not as fast as vectorized but keeps it neat
def var_ext(var,stat):
  sv = pd.Series()
  # index of variable to extract
  idx = stat[stat.loc[:,'VAR']==var].index.tolist()
  # days in each month incl. leap year day
  days = [31,28,31,30,31,30,31,31,30,31,30,31]
  ldays = [31,29,31,30,31,30,31,31,30,31,30,31]
  # convenience variables for years and months
  yr = stat.loc[idx,'YR']
  mo = stat.loc[idx,'MO']
  # loop through every instance of the variable to extract
  for i in idx:
    # number of days in month
    if isleap(yr[i]):
      d = ldays[mo[i]-1]
    else:
      d = days[mo[i]-1]
    # only pull values that correspond to days in the month
    out = stat.loc[i][range(4,4*(d+1),4)] 
    # replace questionable quality values with NaN
    out[pd.np.array(stat.loc[i][range(6,6+4*d,4)].notnull())] = np.nan
    # replace missing values with NaN
    out[out==-9999] = np.nan
    # reindex with proper time stamp
    tm = pd.to_datetime({'year':yr[i],'month':mo[i],'day':range(1,d+1)})
    out = pd.Series(out.values,index=tm)*0.1 # convert to measured units
    sv = pd.concat([sv,out])
  # get full date vector and expand NaNs for missing data
  date1 = pd.to_datetime({'year':yr[idx[0]],'month':mo[idx[0]],'day':[1]})
  date2 = pd.to_datetime({'year':yr[idx[-1]],'month':mo[idx[-1]],'day':[d]})
  all_dates = pd.date_range(start=date1[0],end=date2[0])
  sv = sv.reindex(index=all_dates)
  return sv

### column names for extracting station data from file
wids = [11,4,2,4]+[5,1,1,1]*31
nam = ['VAL','MFL','QFL','SFL']*31
num = range(1,32)*4
num.sort()
val_flg = [nam[i]+str(num[i]) for i in range(len(num))]

# dataframes to save homogenized data
tmax_all = pd.Series()
tmin_all = pd.Series()
prcp_all = pd.Series()

for i,s in enumerate(stations):
  stat = pd.read_fwf(s,widths=wids,
		      header=None,index_col=False,
		      names=['ID','YR','MO','VAR']+val_flg)
  # restrict years
  stat = stat[stat.YR.isin(range(1981,2018))]

  tmax = var_ext('TMAX',stat)
  tmin = var_ext('TMIN',stat)
  prcp = var_ext('PRCP',stat)
  tmax_all = pd.concat([tmax_all,tmax],axis=1)
  tmin_all = pd.concat([tmin_all,tmin],axis=1)
  prcp_all = pd.concat([prcp_all,prcp],axis=1) 
  if i % 25 == 0: print i

# need to strip first column
tmax_all = tmax_all.iloc[:,1:len(tmax_all.columns)+1]
tmax_all.columns = locs.columns
# concatenate and export
out = pd.concat([locs,tmax_all])
# may want to write explicit NaNs in the csv
out.to_csv('tmax.csv')

tmin_all = tmin_all.iloc[:,1:len(tmin_all.columns)+1]
tmin_all.columns = locs.columns
out = pd.concat([locs,tmin_all])
# may want to write explicit NaNs in the csv
out.to_csv('tmin.csv')

prcp_all = prcp_all.iloc[:,1:len(prcp_all.columns)+1]
prcp_all.columns = locs.columns
out = pd.concat([locs,prcp_all])
# may want to write explicit NaNs in the csv
out.to_csv('prcp.csv')
