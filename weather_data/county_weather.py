import shapefile
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import scipy.interpolate as intrp

# include FTP to download census data?
# hi-res counties with interpolated midpoints
shpfile = 'tl_2016_us_county'
usc = shapefile.Reader(shpfile)
fnames = [f[0] for f in usc.fields[1:]]
clocs = pd.DataFrame(usc.records(),columns=fnames)

# retrieve county yield data
ct = pd.read_csv('../crop_data/yld.csv',header=None)

# add GEOID column to yield data
geoid = [ str(ct.iloc[i,2]).zfill(2)+str(ct.iloc[i,3]).zfill(3) 
	  for i in xrange(ct.shape[0]) ]
ct.loc[:,'5'] = pd.Series(geoid,index=ct.index)
ct.columns = ['County','Year','StID','CoID','Yield','GEOID']

# match locations
cmatch = pd.match(ct.GEOID.unique(),clocs.GEOID)
cmatch = cmatch[cmatch != -1] # eliminate failed matches (grouped yields)
ciloc = clocs.loc[cmatch,['GEOID','INTPTLAT','INTPTLON']]
ciloc = ciloc.sort_values(by='GEOID')
# re-index
ciloc.index = range(ciloc.shape[0])

# retrieve daily temperature matrix
tmax = pd.read_csv('tmax.csv')
tmin = pd.read_csv('tmin.csv')
# extract locations (ignore lat/lon at start)
t_locs = tmax.iloc[0:2,1:tmax.shape[1]] 
tl = t_locs.transpose()
tl = np.array(tl.loc[:,[1,0]])
tmax = tmax.iloc[3:tmax.shape[0],1:tmax.shape[1]]
tmin = tmin.iloc[3:tmin.shape[0],1:tmin.shape[1]]

# convert county locations to an array of floats
cl = [ [float(ciloc.loc[i,'INTPTLON']),float(ciloc.loc[i,'INTPTLAT'])] 
       for i in xrange(ciloc.shape[0]) ]
cl = np.array(cl)

# pre-allocate array to save time
svx = np.zeros([tmax.shape[0],len(ciloc.GEOID)])
svi = np.zeros([tmin.shape[0],len(ciloc.GEOID)])

# find missing data points each day
for i in xrange(tmax.shape[0]):
  # TMAX
  tmx = tmax.iloc[i,:]
  pl = pd.notnull(tmx)
  pl = np.array(pl)
  
  # get convex hull
  hull = Delaunay(tl[pl])
  inhull = hull.find_simplex(cl)>0
  
  # 1 point linear interpolator
  lin = intrp.LinearNDInterpolator(tl[pl],tmx[pl])
  lintrp = lin(cl[inhull])
  # 1 point nearest interpolator (outside Qhull)
  near = intrp.NearestNDInterpolator(tl[pl],tmx[pl])
  neartrp = near(cl[~inhull])
  
  # fill empty array to populate the DataFrame
  out = np.zeros(shape=len(cl))
  out[inhull] = lintrp
  out[~inhull] = neartrp
  
  svx[i,:] = out
  
  # TMIN
  tmi = tmin.iloc[i,:]
  pl = pd.notnull(tmi)
  pl = np.array(pl)
  
  # get convex hull
  hull = Delaunay(tl[pl])
  inhull = hull.find_simplex(cl)>0
  
  # 1 point linear interpolator
  lin = intrp.LinearNDInterpolator(tl[pl],tmi[pl])
  lintrp = lin(cl[inhull])
  # 1 point nearest interpolator (outside Qhull)
  near = intrp.NearestNDInterpolator(tl[pl],tmi[pl])
  neartrp = near(cl[~inhull])
  
  # fill empty array to populate the DataFrame
  out = np.zeros(shape=len(cl))
  out[inhull] = lintrp
  out[~inhull] = neartrp
  
  svi[i,:] = out

  if i%500==0: print i

# organize
cou_tmx = pd.DataFrame(data = svx, columns = ciloc.GEOID,
		       index = pd.date_range(start='1981-01-01',end='2016-12-31'))
				  
cou_tmx.index.name = 'Date'
cou_tmx.to_csv('cou_tmx.csv')

cou_tmi = pd.DataFrame(data = svi, columns = ciloc.GEOID,
		       index = pd.date_range(start='1981-01-01',end='2016-12-31'))
				  
cou_tmi.index.name = 'Date'
cou_tmi.to_csv('cou_tmi.csv')
