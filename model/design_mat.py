import pandas as pd
import numpy as np

# state progress
st = pd.read_csv('../crop_data/progress.csv',index_col=[0,1])

st.index.names = ['State','Date']

stp = pd.DataFrame(data={'Veg':st.Planted-st.Silking,'EGF':st.Silking-st.Dough, 
			 'LGF':st.Dough-st.Mature, 'Dry':st.Mature-st.Harvest},
		   columns=['Veg','EGF','LGF','Dry'])

# remove NaN values using EGF column as index
stp = stp[~np.isnan(stp.EGF.values)]

# re-build datetime index
sti = [stp.index[i][0] for i in xrange(stp.shape[0])]
dti = [pd.to_datetime(stp.index[i][1]) for i in xrange(stp.shape[0])]

stp = pd.DataFrame(stp.values,index=[sti,dti],columns=['Veg','EGF','LGF','Dry'])
stp.index.names = ['State','Date']

# State IDs
fips = map(str,[17,18,19,20,21,26,27,29,31,39,46,55])
# State Names
stn = ['ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY','MICHIGAN',
       'MINNESOTA','MISSOURI','NEBRASKA','OHIO','SOUTH DAKOTA','WISCONSIN']

# loop through states to get mean development
sp = list()
# 80s mean development
s80 = list()
# last decade mean development
sld = list()
for s in stn:
  g = stp.loc[s,:].groupby(stp.loc[s,:].index.dayofyear)
  gm = g.mean()
  sp.append(gm)
  # evidence of adaptation
  # 10 yr
  g80 = stp.loc[s,:]['1981':'1990']
  # 5 yr
  #g80 = stp.loc[s,:]['1981':'1985']
  # 15 yr
  #g80 = stp.loc[s,:]['1981':'1995']
  g80m = g80.groupby(g80.index.dayofyear).mean()
  s80.append(g80m)
  # 10 yr
  gld = stp.loc[s,:]['2008':'2017']
  # 5 yr
  #gld = stp.loc[s,:]['2013':'2017'] 
  # 15 yr 
  #gld = stp.loc[s,:]['2003':'2017']
  gldm = gld.groupby(gld.index.dayofyear).mean()
  sld.append(gldm)

# county weather
# tmin
ci = pd.read_csv('../weather_data/cou_tmi.csv',index_col='Date')
ci = pd.DataFrame(ci.values,index=pd.to_datetime(ci.index),columns=ci.columns)
# tmax
cx = pd.read_csv('../weather_data/cou_tmx.csv',index_col='Date')
cx = pd.DataFrame(cx.values,index=pd.to_datetime(cx.index),columns=cx.columns)
# prcp
cp = pd.read_csv('../weather_data/cou_prc.csv',index_col='Date')
cp = pd.DataFrame(cp.values,index=pd.to_datetime(cp.index),columns=cp.columns)


# geoid and year iterators
geoid = list(ci.columns)
yrs = map(str,range(1981,2018))

# calculate GDD
def gdd_calc(ci,cx):
  cit = ci.copy()
  cxt = cx.copy()
  cit[cit<9] = 9
  cit[cit>29] = 29
  cxt[cxt<9] = 9
  cxt[cxt>29] = 29
  gdd = (cit+cxt)/2 - 9
  return gdd

gdd = gdd_calc(ci,cx)

# calculate kdd
def kdd_calc(cx):
  cxt = cx.copy()
  cxt[cxt<29] = 29
  kdd = cxt - 29
  return kdd

kdd = kdd_calc(cx)

# weight matrix for percent of state in each phase on each day
# extract state ID from full GEOID
geo = [g[0:2] for g in geoid]
ndim = list(ci.shape)
ndim.append(4)
wmat = np.zeros(ndim)
wmatm = np.zeros(ndim)
# first and last decade for adaptation
wm80m = np.zeros(ndim)
wmldm = np.zeros(ndim)
for p in range(4):
  for i,f in enumerate(fips):
    m = [f==g for g in geo]
    ci.loc[:,m] 
    # re-index to match weather then reformat to matrix for aggregation
    stt = stp.loc[stn[i],:]
    stt = stt.reindex(ci.index)
    wphas = np.array([stt.iloc[:,p],]*sum(m)).transpose()
    # get locations in array
    idx = [l for l,v in enumerate(m) if v == True]
    wmat[:,idx,p] = wphas
    wphasm = list()
    wp80m = list()
    wpldm = list()
    # index through years of each state's mean development (sp)
    for y in yrs:
      yl = stt[y].shape[0]
      wphasm.append(sp[i].iloc[0:yl,p])
      wp80m.append(s80[i].iloc[0:yl,p])
      wpldm.append(sld[i].iloc[0:yl,p])
    wphasm = np.array([np.concatenate(wphasm),]*sum(m)).transpose()
    wmatm[:,idx,p] = wphasm
    wp80m = np.array([np.concatenate(wp80m),]*sum(m)).transpose()
    wm80m[:,idx,p] = wp80m    
    wpldm = np.array([np.concatenate(wpldm),]*sum(m)).transpose()
    wmldm[:,idx,p] = wpldm    

# put nans into mean development
wmatm[np.isnan(wmat)] = np.nan
wm80m[np.isnan(wmat)] = np.nan
wmldm[np.isnan(wmat)] = np.nan

# design matrix calculated from gdd, kdd, and phase weights
def gkpmat(gdd,kdd,cp,wmat):
  gdd_yr = list()
  kdd_yr = list()
  prc_yr = list()
  for i in range(4):
    tmp = gdd * wmat[:,:,i]/100
    tmp = [tmp[y].sum(skipna=False) for y in yrs]
    df = pd.DataFrame(data=tmp,index = yrs)
    gdd_yr.append(df.transpose().stack())
    tmp = kdd * wmat[:,:,i]/100
    tmp = [tmp[y].sum(skipna=False) for y in yrs]
    df = pd.DataFrame(data=tmp,index = yrs)
    kdd_yr.append(df.transpose().stack())
    tmp = cp * wmat[:,:,i]/100                      
    tmp = [tmp[y].sum(skipna=False) for y in yrs]
    df = pd.DataFrame(data=tmp,index = yrs)
    prc_yr.append(df.transpose().stack())
  out = pd.concat(gdd_yr+kdd_yr+prc_yr,axis=1)
  out.columns = ['Veg_GDD','EGF_GDD','LGF_GDD','Dry_GDD',
		 'Veg_KDD','EGF_KDD','LGF_KDD','Dry_KDD',
		 'Veg_PRC','EGF_PRC','LGF_PRC','Dry_PRC']
  out = out[~np.isnan(out.Veg_GDD.values)] # remove NaNs
  out.index.names = ['geoid','Year']
  # index shenanigans to sync with yield
  out = out.reset_index(level='Year')
  out.Year = pd.to_numeric(out.Year)
  return out

# design matrix
d_mat = gkpmat(gdd,kdd,cp,wmat)
d_mat.index = [d_mat.index,d_mat.Year]

# add in growing season precip and square
PRC = d_mat.Veg_PRC + d_mat.EGF_PRC + d_mat.LGF_PRC
PRC.name = 'PRC'
PRC2 = PRC**2
PRC2.name = 'PRC2'
d_mat = d_mat.join([PRC,PRC2])

# GDD and KDD climatologies

# climatology of daily weather
#ci_clim = np.zeros(ci.shape)
#cx_clim = np.zeros(cx.shape)
#for i,c in enumerate(geoid):
#  # county climatology from grouped object
#  climi = ci[c].groupby(ci[c].index.dayofyear).mean()
#  climx = cx[c].groupby(cx[c].index.dayofyear).mean()
#  # temporary variable to hold county years
#  tmpi = list()
#  tmpx = list()
#  for y in yrs:                        
#    tmpi.append(climi[0:len(ci[y])])
#    tmpx.append(climx[0:len(cx[y])])
#  # collect in array
#  ci_clim[:,i] = np.concatenate(tmpi)
#  cx_clim[:,i] = np.concatenate(tmpx)
## organize in dataframe
#ci_clim = pd.DataFrame(ci_clim, index=ci.index, columns=ci.columns)
#cx_clim = pd.DataFrame(cx_clim, index=cx.index, columns=cx.columns)

# gdd_clim = gdd_calc(ci_clim,cx_clim) 

# kdd_clim = kdd_calc(cx_clim)

# climatology of daily GDD/KDD
gdd_clim = np.zeros(gdd.shape)
kdd_clim = np.zeros(kdd.shape)
prc_clim = np.zeros(cp.shape)
for i,c in enumerate(geoid):
  # county climatology from grouped object
  gclim = gdd[c].groupby(gdd[c].index.dayofyear).mean()
  kclim = kdd[c].groupby(kdd[c].index.dayofyear).mean()
  pclim = cp[c].groupby(cp[c].index.dayofyear).mean()
  # temporary variable to hold county years
  tmpi = list()
  tmpx = list()
  tmpp = list()
  for y in yrs:                        
    tmpi.append(gclim[0:len(ci[y])])
    tmpx.append(kclim[0:len(cx[y])])
    tmpp.append(pclim[0:len(cp[y])])
  # collect in array
  gdd_clim[:,i] = np.concatenate(tmpi)
  kdd_clim[:,i] = np.concatenate(tmpx)
  prc_clim[:,i] = np.concatenate(tmpp)
# organize in dataframe
gdd_clim = pd.DataFrame(gdd_clim, index=ci.index, columns=ci.columns)
kdd_clim = pd.DataFrame(kdd_clim, index=cx.index, columns=cx.columns)
prc_clim = pd.DataFrame(prc_clim, index=cp.index, columns=cp.columns)

# climatology of weather and development
d_clim = gkpmat(gdd_clim,kdd_clim,prc_clim,wmatm)
d_clim.index = [d_clim.index,d_clim.Year]
# add in growing season precip and square
PRC = d_clim.Veg_PRC + d_clim.EGF_PRC + d_clim.LGF_PRC
PRC.name = 'PRC'
PRC2 = PRC**2
PRC2.name = 'PRC2'
d_clim = d_clim.join([PRC,PRC2])
# climatology of weather w/ variable development
d_dev = gkpmat(gdd_clim,kdd_clim,prc_clim,wmat)
d_dev.index = [d_dev.index,d_dev.Year]
# add in growing season precip and square
PRC = d_dev.Veg_PRC + d_dev.EGF_PRC + d_dev.LGF_PRC
PRC.name = 'PRC'
PRC2 = PRC**2
PRC2.name = 'PRC2'
d_dev = d_dev.join([PRC,PRC2])
# variable weather w/ fixed development
d_wea = gkpmat(gdd,kdd,cp,wmatm)
d_wea.index = [d_wea.index,d_wea.Year]
# add in growing season precip and square
PRC = d_wea.Veg_PRC + d_wea.EGF_PRC + d_wea.LGF_PRC
PRC.name = 'PRC'
PRC2 = PRC**2
PRC2.name = 'PRC2'
d_wea = d_wea.join([PRC,PRC2])
# variable weather w/80s fixed development
d_w80 = gkpmat(gdd,kdd,cp,wm80m)
d_w80.index = [d_w80.index,d_w80.Year]
# add in growing season precip and square
PRC = d_w80.Veg_PRC + d_w80.EGF_PRC + d_w80.LGF_PRC
PRC.name = 'PRC'
PRC2 = PRC**2
PRC2.name = 'PRC2'
d_w80 = d_w80.join([PRC,PRC2])
# variable weather w/last decade's fixed development
d_wld = gkpmat(gdd,kdd,cp,wmldm)
d_wld.index = [d_wld.index,d_wld.Year]
# add in growing season precip and square
PRC = d_wld.Veg_PRC + d_wld.EGF_PRC + d_wld.LGF_PRC
PRC.name = 'PRC'
PRC2 = PRC**2
PRC2.name = 'PRC2'
d_wld = d_wld.join([PRC,PRC2])

# getting yield data synced up with weather
yld = pd.read_csv('../crop_data/yld.csv',
		  header=None,names=['Cou','Year','StID','CouID','Yld'])
# remove joint county estimates
yld = yld[yld.CouID != 998]
# remove Ste. Genevieve county (absent from weather data)
yld = yld[yld.Cou != 'STE. GENEVIEVE']
# build a common index
geoid = [str(yld.StID[i]).zfill(2)+str(yld.CouID[i]).zfill(3) for i in yld.index]
yld.index = geoid

# remove irrigated counties
harv = pd.read_csv('../crop_data/harv_area.csv',
		  header=None,names=['Cou','Year','StID','CouID','Har'])
geoid = [str(harv.StID[i]).zfill(2)+str(harv.CouID[i]).zfill(3) for i in harv.index]
harv.index = geoid
#idx = harv.Year != 2012
#harv = harv.loc[idx]

har_av = pd.DataFrame(harv['Har'].groupby(harv.index).mean())

irr = pd.read_csv('../crop_data/irr_area.csv',
		  header=None,names=['Cou','Year','StID','CouID','Irr'])
geoid = [str(irr.StID[i]).zfill(2)+str(irr.CouID[i]).zfill(3) for i in irr.index]
irr.index = geoid
#idx = irr.Year != 2012
#irr = irr.loc[idx]

irr_av = pd.DataFrame(irr['Irr'].groupby(irr.index).mean())

har_irr = har_av.join(irr_av)

har_irr['Irr_perc'] = har_irr.Irr/har_irr.Har

# remove these indices from the yld DataFrame
irr_idx = har_irr[har_irr['Irr_perc']>0.1].index

yld = yld.drop(irr_idx)

# need to remove counties W of -100, in the UP, and yield records shorter than 10 (25)
# years 
# michigan UP county fips
MIUPfips = map(str,[26033,26097,26095,26153,26003,26041,26103,26109,26043,
	    26071,26013,26061,26131,26053]) # already removed 26083
yld = yld.drop(MIUPfips)

# counties W of -100
import shapefile

shpfile = '../weather_data/tl_2016_us_county'
usc = shapefile.Reader(shpfile)
fnames = [f[0] for f in usc.fields[1:]]
clocs = pd.DataFrame(usc.records(),columns=fnames)
clocs.index = clocs.GEOID
clocs.INTPTLON = map(float,clocs.INTPTLON)
idx = clocs[clocs.INTPTLON<-100]
yld = yld.drop(idx.index)

# counties with twenty-five or fewer years
ugeo = yld.index.unique()
cy = [yld.loc[l,'Yld'].size for l in ugeo]
locy = [c<25 for c in cy]
yld = yld.drop(ugeo[np.array(locy)])

# set up indices to remove years
yld.index = [yld.index,yld.Year]
yld.index.names = ['geoid','Year']

# match single index
yd_mat = pd.DataFrame()
# convert yield to tonnes/hectare
# 1 bu/ac = .0628 t/ha
for g in np.unique(yld.index.get_level_values(0)):
  tmp = d_mat.loc[g].join(yld.loc[g].Yld*0.0628)
  tmp['geoid'] = g
  tmp = tmp[~np.isnan(tmp.Yld)]
  # remove mean from each counties weather variables
  m = tmp[tmp.columns[1:15]].mean()
  tmp[tmp.columns[1:15]] = tmp[tmp.columns[1:15]] - m
  tmp.Year = tmp.Year - 1981 # count 1981 as year 0
  yd_mat = yd_mat.append(tmp)

# reindex
yd_mat.index = [yd_mat.geoid,yd_mat.Year]

# add lat/lon
cll = pd.DataFrame(clocs.loc[:,['INTPTLAT','INTPTLON']])
lat = [x[1:] for x in cll.loc[:,'INTPTLAT']]
cll.INTPTLAT = map(float,lat)
cll.columns = ['lat','lon']
cll.index.name = 'geoid'

yd_mat = yd_mat.join(cll)

# weighted mean daily GDD and KDD at each station each year
def gkpmean_mat(gdd,kdd,cp,wmat):
  gdd_yr = list()
  kdd_yr = list()
  prc_yr = list()
  yrs = map(str,range(1981,2018))
  geoid = list(ci.columns)
  for i in range(4):
    wtmp = pd.DataFrame(wmat[:,:,i],index=gdd.index)
    tmp = [np.average(gdd[y],axis=0,weights=wtmp[y]) for y in yrs]
    df = pd.DataFrame(data=tmp,index = yrs,columns=geoid)
    gdd_yr.append(df.transpose().stack())
    tmp = [np.average(kdd[y],axis=0,weights=wtmp[y]) for y in yrs]
    df = pd.DataFrame(data=tmp,index = yrs,columns=geoid)
    kdd_yr.append(df.transpose().stack())
    tmp = [np.average(cp[y],axis=0,weights=wtmp[y]) for y in yrs]
    df = pd.DataFrame(data=tmp,index = yrs,columns=geoid)
    prc_yr.append(df.transpose().stack())    
  out = pd.concat(gdd_yr+kdd_yr+prc_yr,axis=1)
  out.columns = ['Veg_GDD','EGF_GDD','LGF_GDD','Dry_GDD',
  		 'Veg_KDD','EGF_KDD','LGF_KDD','Dry_KDD',
		 'Veg_PRC','EGF_PRC','LGF_PRC','Dry_PRC']
  out.index.names = ['geoid','Year']
  # index shenanigans to sync with yield
  out = out.reset_index(level='Year')
  out.Year = pd.to_numeric(out.Year)
  return out

d_gkpmeans = gkpmean_mat(gdd,kdd,cp,wmat)
d_gkpmeans.index = [d_gkpmeans.index,d_gkpmeans.Year]

stp.to_csv('stp.csv')
yd_mat.to_csv('yd_mat.csv')
d_mat.to_csv('d_mat.csv')
d_clim.to_csv('d_clim.csv')
d_dev.to_csv('d_dev.csv')
d_wea.to_csv('d_wea.csv')
d_gkpmeans.to_csv('d_gkpmeans.csv')

# change underscore when using different averaging periods
d_w80.to_csv('d_w80_10.csv')
d_wld.to_csv('d_wld_10.csv')
