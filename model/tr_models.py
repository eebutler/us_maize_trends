import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from numpy.linalg import LinAlgError

stp = pd.read_csv('stp.csv',index_col=[0,1])
# convert stp index to DatetimeIndex
idx = stp.index.get_level_values(1)
idx = pd.to_datetime(idx,infer_datetime_format=True)
stp.index = [stp.index.get_level_values(0),idx]
stp.index.names = ['State','Date']

# primary yield/data matrix
yd_mat = pd.read_csv('yd_mat.csv',index_col=[0,1])
# straighten out the column names
col = yd_mat.columns.values
col[0] = 'Year'
col[-1] = 'geoid'
yd_mat.columns = col
# convert geoid to string to access both state and county components
yd_mat.geoid = map(str,yd_mat.geoid)

# weather DFs
d_mat = pd.read_csv('d_mat.csv',index_col=[0,1])
col = d_mat.columns.values
col[0] = 'Year'
d_mat.columns = col
d_clim = pd.read_csv('d_clim.csv',index_col=[0,1])
d_clim.columns = col
d_dev = pd.read_csv('d_dev.csv',index_col=[0,1])
d_dev.columns = col
d_wea = pd.read_csv('d_wea.csv',index_col=[0,1])
d_wea.columns = col
d_gkmeans = pd.read_csv('d_gkmeans.csv',index_col=[0,1])
d_gkmeans.columns = col
d_w80 = pd.read_csv('d_w80_10.csv',index_col=[0,1])
d_w80.columns = col
d_wld = pd.read_csv('d_wld_10.csv',index_col=[0,1])
d_wld.columns = col
d_w80_5 = pd.read_csv('d_w80_5.csv',index_col=[0,1])
d_w80_5.columns = col
d_wld_5 = pd.read_csv('d_wld_5.csv',index_col=[0,1])
d_wld_5.columns = col
d_w80_15 = pd.read_csv('d_w80_15.csv',index_col=[0,1])
d_w80_15.columns = col
d_wld_15 = pd.read_csv('d_wld_15.csv',index_col=[0,1])
d_wld_15.columns = col

# Calculate bootstrapped uncertainties? (time consuming 8+ hours)
boot_ests = False

# state names
stnms = np.unique(stp.index.get_level_values(0))

# State ID numbers 
fips = map(str,[17,18,19,20,21,26,27,29,31,39,46,55])

# State Names
stn = ['ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY','MICHIGAN',
       'MINNESOTA','MISSOURI','NEBRASKA','OHIO','SOUTH DAKOTA','WISCONSIN']

# set up the regression covariates
cols = list(yd_mat.columns[0:9])
# omit drydown
cols = cols[0:4] + cols[5:8]
form = 'Yld ~ C(geoid)+'+'+'.join(cols)
mod = smf.ols(formula=form,data=yd_mat).fit()
# get county level R^2
cr2 = []
fva = mod.fittedvalues

beta = mod.params[-6:]
tr = mod.params[-7]

# individual county fits
ct = yd_mat.index.get_level_values(0).unique()
for c in ct:
  r2 = np.corrcoef(fva.loc[c],yd_mat.loc[c].Yld)[0,1]**2  
  cr2.append(r2)

cr2 = pd.DataFrame(cr2,index=ct)
cr2.to_csv('cr2.csv')

beta_boot = np.zeros([6,1000])
tre_boot = np.zeros(1000)

if boot_ests:
  sh = yd_mat.shape[0]
  # bootstrap on county years
  i = 0
  while i < 1000:
    try:
      ind = np.random.randint(sh,size=sh)
      out = smf.ols(formula=form,data=yd_mat.iloc[ind]).fit()
      beta_boot[:,i] = out.params[-6:]
      tre_boot[i] = out.params[-7]
      i += 1
      if i%50==0: print i
    except LinAlgError:
      continue
  # betas
  tmp = np.sort(beta_boot,axis=1)
  betas = pd.DataFrame(tmp.T,columns=cols[1:7])
  betas.index = range(1,1001)
  # 95% CI
  # trend
  tres = pd.DataFrame(np.sort(tre_boot),index=range(1,1001),columns=['trend'])
  betas.to_csv('beta_boot.csv')
  tres.to_csv('trend_boot.csv')
else:
  betas = pd.read_csv('beta_boot.csv',index_col=0)
  tres = pd.read_csv('trend_boot.csv',index_col=0)

# to get 95% CI
#beta95 = np.array(betas)
#beta95 = np.sort(test,axis=0)
#beta95[[24,974],:]

## get mean values Duration for each year and state
# remove Year from cols
cols = cols[1:7]
# total "days" in each phase in each year for a state
mdev = pd.DataFrame()
for s in stnms:
  tmp = stp.loc[s].groupby(pd.TimeGrouper("A")).sum()/100
  tmp.index = [[s]*len(tmp),tmp.index.year]
  mdev = mdev.append(tmp)

# get planted area to weight GDD/KDD values
pl = pd.read_csv('../crop_data/pl_area.csv',
		  header=None,names=['Cou','Year','StID','CouID','Area'])

geoid = [str(pl.StID[i]).zfill(2)+str(pl.CouID[i]).zfill(3) for i in pl.index]
pl.index = [map(int,geoid),pl.Year]
pl.index.names = ['geoid','Year']

# calculate state level weighted averages of GDD/KDD by phase
def st_summary(dmat):
  yld = pd.DataFrame(yd_mat.Yld)
  yld.index = [yld.index.get_level_values(0),yld.index.get_level_values(1)+1981] 
  yda_mat = pd.DataFrame()
  for g in np.unique(yld.index.get_level_values(0)):
    tmp = dmat.loc[g].join(yld.loc[g].Yld)
    tmp['geoid'] = str(g)
    tmp = tmp[~np.isnan(tmp.Yld)]
    tmp = tmp.join(pl.loc[g].Area)
    yda_mat = yda_mat.append(tmp)
    
  yda_st = [g[0:2] for g in yda_mat.geoid]

  yda_mat.index = [yda_st,yda_mat.Year]

  # dictionary for applying weighted average to each column
  # needed due to calling apply on two columns... more elegant solution?
  wa_dict = {'Veg_GDD':lambda x: np.average(x.Veg_GDD, weights=x.Area),
	     'EGF_GDD':lambda x: np.average(x.EGF_GDD, weights=x.Area),
	     'LGF_GDD':lambda x: np.average(x.LGF_GDD, weights=x.Area),
	     'Dry_GDD':lambda x: np.average(x.Dry_GDD, weights=x.Area),
	     'Veg_KDD':lambda x: np.average(x.Veg_KDD, weights=x.Area), 
             'EGF_KDD':lambda x: np.average(x.EGF_KDD, weights=x.Area),
             'LGF_KDD':lambda x: np.average(x.LGF_KDD, weights=x.Area),
             'Dry_KDD':lambda x: np.average(x.Dry_KDD, weights=x.Area)}

  # column indices for looping
  cols = yda_mat.columns[1:9]
  
  st_summ = pd.DataFrame()
  # loop to get weighted mean GDD/KDD and append mean duration
  for i,s in enumerate(np.unique(yda_st)):
    g = yda_mat.loc[s].groupby(yda_mat.loc[s].index)
    tmp = pd.DataFrame()
    for c in cols:
      out = g.apply(wa_dict[c])
      out.name = c
      tmp = tmp.append(out)
    tmp = tmp.transpose()
    tmp = tmp.join(mdev.loc[stnms[i]])
    tmp.index = [[stnms[i]]*tmp.shape[0],tmp.index]
    st_summ = st_summ.append(tmp)
  return st_summ

# but duration also depends on the temperature experienced...
# restrict to KDD for consistency with B&H (2015)
st_means = st_summary(d_gkmeans)

kd_sens = np.zeros([len(stnms),4])
kd_fit = np.zeros([len(stnms),4])
kd_sens_boot = np.zeros([len(stnms),4,1000])
for i,s in enumerate(stnms):
  sz = st_means.loc[s].shape[0]
  idx = np.random.randint(sz,size=[sz,1000])
  for j in range(4):
    Y = st_means.loc[s].iloc[:,8+j]
    X = st_means.loc[s].iloc[:,4+j]
    X = sm.add_constant(X)
    b = sm.OLS(Y,X)
    kd_sens[i,j] = b.fit().params[1]
    kd_fit[i,j] = b.fit().rsquared
    if boot_ests:
      for k in range(1000):
        b = sm.OLS(Y.iloc[idx[:,k]],X.iloc[idx[:,k],:])
        kd_sens_boot[i,j,k] = b.fit().params[1]
      kd_sens_boot[i,j,:] = np.sort(kd_sens_boot[i,j,:])

kd_sens = pd.DataFrame(kd_sens,columns=['Veg','EGF','LGF','Dry'],index=stnms)
kd_fit = pd.DataFrame(kd_fit,columns=['Veg','EGF','LGF','Dry'],index=stnms)
kd_fit = kd_fit.round(decimals=3)*100
#kd_fit.to_latex('duration_fit.txt')

# state mean g/kdd
st_gr = st_means.groupby(level=0)
st_grm = st_gr.mean()

# dict to connect fips with state names
fsd = dict(zip(fips,stnms))

# all county geoids
ct = d_gkmeans.index.get_level_values(0).unique()

gk_veg_anoms = pd.DataFrame()
for c in ct:
  # get the state name from the geoid
  s = fsd[str(c)[0:2]]
  anoms = d_gkmeans.loc[c,cols] - st_grm.loc[s,cols]
  y = d_gkmeans.loc[c].shape[0]
  # day anomaly from yearly KDD anomaly and KDD-Day sensitivity
  dys = anoms.iloc[:,3:6].values*np.array([kd_sens.loc[s,['Veg','EGF','LGF']]]*y)
  # collect anomalies in each phase 
  tmp = np.tile(dys,2)*([np.array(st_grm.loc[s,cols])]*y)
  tmp = pd.DataFrame(tmp,columns=cols)
  gk_veg_anoms = gk_veg_anoms.append(tmp)

gk_veg_anoms.index = d_gkmeans.index

# subtract anomaly from dev and add to wea
d_deva = d_dev.loc[:,cols] - gk_veg_anoms
d_deva['Year'] = d_dev.Year
d_weaa = d_wea.loc[:,cols] + gk_veg_anoms 
d_weaa['Year'] = d_wea.Year
#d_w80a = d_w80.loc[:,cols] + gk_veg_anoms 
#d_w80a['Year'] = d_w80.Year
#d_wlda = d_wld.loc[:,cols] + gk_veg_anoms 
#d_wlda['Year'] = d_wld.Year

# unique county geoids with yields
ct = yd_mat.index.get_level_values(0).unique()

def ctrends(dmat,da_or_wa):
  ctr = pd.DataFrame(columns=cols)
  ctr_b = np.zeros([len(ct),6,1000]) 
  for k,c in enumerate(ct):
    # trends in each counties predictors
    # restrict to years with yields
    X = yd_mat.loc[c].Year
    Y = dmat.loc[c,cols].iloc[X.values]
    # line up indices
    Y.index = X.index
    cb = np.zeros(len(cols)) 
    for i,p in enumerate(cols):
      X = sm.add_constant(X)
      b = sm.OLS(Y.loc[:,p],X).fit().params
      cb[i] = b[1]
    # assemble the county level trends
    tmp = pd.DataFrame(cb,index=cols,columns=[c])
    ctr = ctr.append(tmp.T)
    if boot_ests:
      print c
      for i in range(1000):
        # random years in each county
        ridx = np.random.randint(len(X),size=len(X))
        Xi = X.iloc[ridx]
        if da_or_wa==False:
	  Yi = Y.iloc[ridx]
	else:
	  # also randomize daylength-Temp. sensitivity, identical to above
          si = np.where(np.array(fips)==str(c)[0:2])[0][0]
          bi = np.random.randint(1000,size=3)
          b = [kd_sens_boot[si,r,bi[r]] for r in range(3)]
          anoms = d_gkmeans.loc[c,cols] - st_grm.loc[s,cols]  
          y = d_gkmeans.loc[c].shape[0]
          dys = anoms.iloc[:,3:6].values*np.tile(b,[y,1])
          gka = np.tile(dys,2)*([np.array(st_grm.loc[s,cols])]*y)
	  if da_or_wa == 'da':
	    Yi = Y.iloc[ridx] - gka[ridx]
	  else:
	    Yi = Y.iloc[ridx] + gka[ridx]
        for j,p in enumerate(cols):
          b = sm.OLS(Yi.loc[:,p],Xi).fit().params
          ctr_b[k,j,i] = b[1]

  return [ctr,ctr_b]

ctr,ctr_b = ctrends(d_mat,False)
ctr_d,ctr_d_b = ctrends(d_dev,False)
ctr_w,ctr_w_b = ctrends(d_wea,False)
ctr_da,ctr_da_b = ctrends(d_deva,'da')
ctr_wa,ctr_wa_b = ctrends(d_weaa,'wa')

ctr.to_csv('ctr.csv')
ctr_d.to_csv('ctr_d.csv')
ctr_w.to_csv('ctr_w.csv')
ctr_da.to_csv('ctr_da.csv')
ctr_wa.to_csv('ctr_wa.csv')

# keep as csv with slices as comments 
# http://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file - just made it a function
if boot_ests:
  def ndwrite(fname,data):
    with file(fname+'.csv','w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(data.shape))
      for dslice in data:
        np.savetxt(outfile,dslice,delimiter=',')
        outfile.write('# New Slice\n')
 
  ndwrite('ctr_b',ctr_b)
  ndwrite('ctr_d_b',ctr_d_b)
  ndwrite('ctr_w_b',ctr_w_b)
  ndwrite('ctr_da_b',ctr_da_b)
  ndwrite('ctr_wa_b',ctr_wa_b)
else:
  cts = len(ct)
  ctr_b = np.loadtxt('ctr_b.csv',delimiter=',').reshape(cts,6,1000)
  ctr_d_b = np.loadtxt('ctr_d_b.csv',delimiter=',').reshape(cts,6,1000)
  ctr_w_b = np.loadtxt('ctr_w_b.csv',delimiter=',').reshape(cts,6,1000)
  ctr_da_b = np.loadtxt('ctr_da_b.csv',delimiter=',').reshape(cts,6,1000)
  ctr_wa_b = np.loadtxt('ctr_wa_b.csv',delimiter=',').reshape(cts,6,1000)

# expand beta estimates to each county number of counties in each state
cb = np.tile(beta,[ctr.shape[0],1])
cbetas = pd.DataFrame(cb,columns=cols,index=ctr.index)
cbetas.to_csv('cbetas.csv')
# get average planted area in each county for aggregating up to whole belt area
plA = pl.groupby(pl.index.get_level_values(0))['Area'].mean()
plA = plA[ctr.index]

# county yield trends
y_tr = ctr*cbetas
y_trd = ctr_d*cbetas
y_trw = ctr_w*cbetas
y_trda = ctr_da*cbetas
y_trwa = ctr_wa*cbetas

# isolate a state (i.e. kentucky)
stidx = [s[0:2] for s in map(str,ct)]
ky = np.in1d(stidx,'21')


# mean yield trend from development shifts
ytmd = np.average(y_trd.sum(axis=1),weights=plA)
ytdp = ytmd/tr

ytmdf = [np.average(y_trd.iloc[:,i],weights=plA) for i in range(6)]
ytmdfp = ytmdf/tr

# with plant plasticity
ytmda = np.average(y_trda.sum(axis=1),weights=plA)
ytdap = ytmda/tr

ytmdaf = [np.average(y_trda.iloc[:,i],weights=plA) for i in range(6)]
ytmdafp = ytmdaf/tr

# mean yield trend from climate
ytmw = np.average(y_trw.sum(axis=1),weights=plA)
ytwp = ytmw/tr

ytmwf = [np.average(y_trw.iloc[:,i],weights=plA) for i in range(6)]
ytmwfp = ytmwf/tr

# with plant plasticity
ytmwa = np.average(y_trwa.sum(axis=1),weights=plA)
ytwap = ytmwa/tr

ytmwaf = [np.average(y_trwa.iloc[:,i],weights=plA) for i in range(6)]
ytmwafp = ytmwaf/tr

# mean yield trend without separating effects
ytm = np.average(y_tr.sum(axis=1),weights=plA)
ytp = ytm/tr

ytmf = [np.average(y_tr.iloc[:,i],weights=plA) for i in range(6)]
ytmfp = ytmf/tr

# planting area weighted
# bootstrap gdd/kdd and yield trends estimates

trpA = [np.average(ctr.iloc[:,i],weights=plA) for i in range(6)]
trdpA = [np.average(ctr_d.iloc[:,i],weights=plA) for i in range(6)]
trwpA = [np.average(ctr_w.iloc[:,i],weights=plA) for i in range(6)]
trdapA = [np.average(ctr_da.iloc[:,i],weights=plA) for i in range(6)]
trwapA = [np.average(ctr_wa.iloc[:,i],weights=plA) for i in range(6)]

def sort_boot(ctr_nd):
  sortb = ctr_nd 
  for c in range(len(ct)):
    sortb[c,:,:].sort(axis=1)

  return sortb

tr_b = sort_boot(ctr_b)
tr_d_b = sort_boot(ctr_d_b)
tr_w_b = sort_boot(ctr_w_b)
tr_da_b = sort_boot(ctr_da_b)
tr_wa_b = sort_boot(ctr_wa_b)

tr95 = [[np.average(tr_b[:,i,24],weights=plA),
          np.average(tr_b[:,i,974],weights=plA)] for i in range(6)]
trd95 = [[np.average(tr_d_b[:,i,24],weights=plA),
          np.average(tr_d_b[:,i,974],weights=plA)] for i in range(6)]
trw95 = [[np.average(tr_w_b[:,i,24],weights=plA),
          np.average(tr_w_b[:,i,974],weights=plA)] for i in range(6)]
trda95 = [[np.average(tr_da_b[:,i,24],weights=plA),
           np.average(tr_da_b[:,i,974],weights=plA)] for i in range(6)]
trwa95 = [[np.average(tr_wa_b[:,i,24],weights=plA),
	    np.average(tr_wa_b[:,i,974],weights=plA)] for i in range(6)]

# convert into yield trends
def cyboot(ctr_nd):
  ytr = np.zeros([len(ct),6,1000]) 
  for i in range(1000):
    ridx = np.random.randint(1000)
    ridxb = np.random.randint(1000)
    for c in range(len(ct)):
      ytr[c,:,i] = ctr_nd[c,:,ridx]*betas.iloc[ridxb,:].values
  for c in range(len(ct)):
    ytr[c,:,:].sort(axis=1)

  return ytr

ytr_b = cyboot(ctr_b)
ytr_d_b = cyboot(ctr_d_b)
ytr_w_b = cyboot(ctr_w_b)
ytr_da_b = cyboot(ctr_da_b)
ytr_wa_b = cyboot(ctr_wa_b)

ytr95 = [[np.average(ytr_b[:,i,24],weights=plA),
	   np.average(ytr_b[:,i,974],weights=plA)] for i in range(6)]
ytrd95 = [[np.average(ytr_d_b[:,i,24],weights=plA),
	   np.average(ytr_d_b[:,i,974],weights=plA)] for i in range(6)]
ytrw95 = [[np.average(ytr_w_b[:,i,24],weights=plA),
	   np.average(ytr_w_b[:,i,974],weights=plA)] for i in range(6)]
ytrda95 = [[np.average(ytr_da_b[:,i,24],weights=plA),
	    np.average(ytr_da_b[:,i,974],weights=plA)] for i in range(6)]
ytrwa95 = [[np.average(ytr_wa_b[:,i,24],weights=plA),
	    np.average(ytr_wa_b[:,i,974],weights=plA)] for i in range(6)]

# trends in start and end date
pcols = ['Veg_st','EGF_st','LGF_st','Dry_st','Veg_en','EGF_en','LGF_en','Dry_en']
st_all = pd.DataFrame()
# First and Last days of each phase and trend
for i,s in enumerate(stnms):
  yrs = np.unique(stp.loc[s].index.year)
  yrs = map(str,yrs)
  tmp = np.zeros([len(yrs),8])
  for i,y in enumerate(yrs):
    out = stp.loc[s].loc[y].apply(np.nonzero)
    dmin_max = [np.min(o) for o in out] + [np.max(o) for o in out]
    tmp[i,:] = dmin_max
  tmp = pd.DataFrame(tmp,index=[[s]*len(yrs),yrs],columns=pcols)
  st_all = st_all.append(tmp)

# add total duration
mdev.index = [mdev.index.get_level_values(0),map(str,mdev.index.get_level_values(1))]
st_all = st_all.join(mdev)

def tr_boot(st_means_df,c=12):
  phs_tr = np.zeros([len(stnms),c])
  boot_ar = np.zeros([len(stnms),c,1000])
  
  for i,s in enumerate(stnms):
    yrs = np.array(map(int,st_means_df.loc[s].index))
    yrs = sm.add_constant(yrs)
    # use the same random index for all predictors to preserve temporal correlation
    idx = np.random.randint(yrs.shape[0],size=[yrs.shape[0],1000])
    print s 
    for j in range(c):
      Y = st_means_df.loc[s].iloc[:,j]
      b = sm.OLS(Y,yrs)
      # to extract the trend
      phs_tr[i,j] = b.fit().params[1]
      for k in range(1000):
        # to get draws with replacement
        b = sm.OLS(Y.iloc[idx[:,k]],yrs[idx[:,k],:])
        boot_ar[i,j,k] = b.fit().params[1]
      # sort bootstrapped estimates
      boot_ar[i,j,:] = np.sort(boot_ar[i,j,:])
  return (phs_tr,boot_ar)

ph_tr, boot = tr_boot(st_all)

# area weighted date shifts
tmp = map(str,plA.index)
tmp = [t[0:2] for t in tmp]
plAs = pd.DataFrame(plA,columns=['Area'])
plAs['st'] = tmp
tot_ar = plAs.groupby(plAs.st)['Area'].sum()

ph_tr = pd.DataFrame(ph_tr,columns=st_all.columns,index=stnms)
ph_trwa = [np.average(ph_tr.iloc[:,i],weights=tot_ar) for i in range(12)]
ph_trwa = np.round(ph_trwa,decimals=3)

phtr95 = [[boot[:,i,24].mean(),boot[:,i,974].mean()] for i in range(12)]

# all region average by planted area (to allow for weather)
# including using 80's and last decade mean development dates
def ar_summary(dmat):
  yld = pd.DataFrame(yd_mat.Yld)
  yld.index = [yld.index.get_level_values(0),yld.index.get_level_values(1)+1981] 
  yda_mat = pd.DataFrame()
  for g in np.unique(yld.index.get_level_values(0)):
    tmp = dmat.loc[g].join(yld.loc[g].Yld)
    tmp['geoid'] = str(g)
    tmp = tmp[~np.isnan(tmp.Yld)]
    tmp = tmp.join(pl.loc[g].Area)
    yda_mat = yda_mat.append(tmp)
    
  # dictionary for applying weighted average to each column
  # needed due to calling apply on two columns... more elegant solution?
  wa_dict = {'Veg_GDD':lambda x: np.average(x.Veg_GDD, weights=x.Area),
	     'EGF_GDD':lambda x: np.average(x.EGF_GDD, weights=x.Area),
	     'LGF_GDD':lambda x: np.average(x.LGF_GDD, weights=x.Area),
	     'Veg_KDD':lambda x: np.average(x.Veg_KDD, weights=x.Area), 
             'EGF_KDD':lambda x: np.average(x.EGF_KDD, weights=x.Area),
             'LGF_KDD':lambda x: np.average(x.LGF_KDD, weights=x.Area)}

  ar_summ = pd.DataFrame()
  g = yda_mat.groupby(yda_mat.index)
  tmp = pd.DataFrame()
  for c in cols:
    out = g.apply(wa_dict[c])
    out.name = c
    tmp = tmp.append(out)
  tmp = tmp.transpose()
  ar_summ = ar_summ.append(tmp)
  return ar_summ

ar_gk = ar_summary(d_mat)
ar_gk_clim = ar_summary(d_clim)
ar_gk_dev = ar_summary(d_dev)
ar_gk_wea = ar_summary(d_wea)
ar_gk_deva = ar_summary(d_deva)
ar_gk_weaa = ar_summary(d_weaa)
ar_gk_w80 = ar_summary(d_w80)
ar_gk_wld = ar_summary(d_wld)
ar_gk_w80_5 = ar_summary(d_w80_5)
ar_gk_wld_5 = ar_summary(d_wld_5)
ar_gk_w80_15 = ar_summary(d_w80_15)
ar_gk_wld_15 = ar_summary(d_wld_15)

# expand beta values for whole region average
ybeta = pd.DataFrame([beta.values]*36,index=range(1981,2017),columns=cols)

# convert to yield
yld_pcomp = (ar_gk - ar_gk_clim)*ybeta
yld_compda = (ar_gk_deva - ar_gk_clim)*ybeta
yld_compwa = (ar_gk_weaa - ar_gk_clim)*ybeta
yld_compw80 = (ar_gk_w80 - ar_gk_clim)*ybeta
yld_compwld = (ar_gk_wld - ar_gk_clim)*ybeta
yld_compw80_5 = (ar_gk_w80_5 - ar_gk_clim)*ybeta
yld_compwld_5 = (ar_gk_wld_5 - ar_gk_clim)*ybeta
yld_compw80_15 = (ar_gk_w80_15 - ar_gk_clim)*ybeta
yld_compwld_15 = (ar_gk_wld_15 - ar_gk_clim)*ybeta

# difference in yield from last decade compared to 80's development
dur_diff = yld_compwld.sum(axis=1) - yld_compw80.sum(axis=1)
dur_diff_5 = yld_compwld_5.sum(axis=1) - yld_compw80_5.sum(axis=1)
dur_diff_15 = yld_compwld_15.sum(axis=1) - yld_compw80_15.sum(axis=1)

dur_tr = np.polyfit(dur_diff.index,dur_diff,1)[0]
dur_tr_5 = np.polyfit(dur_diff.index,dur_diff_5,1)[0]
dur_tr_15 = np.polyfit(dur_diff.index,dur_diff_15,1)[0]

# combine yield components in GDD/KDD and development/weather
ydaG = yld_compda.loc[:,['Veg_GDD','EGF_GDD','LGF_GDD']].sum(axis=1)
ydaK = yld_compda.loc[:,['Veg_KDD','EGF_KDD','LGF_KDD']].sum(axis=1)
ywaG = yld_compwa.loc[:,['Veg_GDD','EGF_GDD','LGF_GDD']].sum(axis=1)
ywaK = yld_compwa.loc[:,['Veg_KDD','EGF_KDD','LGF_KDD']].sum(axis=1)

# area average yield
pl.index = [pl.index.get_level_values(0),pl.Year-1981]
tmp = yd_mat.join(pl.Area)
g = tmp.groupby(tmp.Year)
yavg = g.apply(lambda x: np.average(x.Yld,weights=x.Area))
yavg.index = yavg.index+1981

yld_comb = pd.DataFrame([yavg,ydaG,ydaK,ywaG,ywaK]).T
yld_comb.columns = ['yld','Dev_GDD','Dev_KDD','Wea_GDD','Wea_KDD']
yld_comb.to_csv('yld_comb.csv')

# best way to report trends?
tr_tmp = list()
for c in yld_comb.columns:
 tr_tmp.append(np.polyfit(yld_comb.index.values,yld_comb.loc[:,c],1)[0])

# trend components
tr_all = [tr,np.sum(ytmwaf[0:3]),np.sum(ytmwaf[3:6]),np.sum(ytmdaf[0:3]),np.sum(ytmdaf[3:6])]

tr_comb = [tr,np.sum(ytmwaf),np.sum(ytmdaf)]

# compound trend uncertainty
def comp_cyboot(ctr_nd):
  ytr = np.zeros([len(ct),6,1000])
  cytr = np.zeros([len(ct),1000])
  for i in range(1000): 
    for c in range(len(ct)):
      for p in range(6):
	ridx = np.random.randint(1000)
	ytr[c,p,i] = ctr_nd[c,p,ridx]
      cytr[c,i] = np.sum(ytr[c,:,i])
  for c in range(len(ct)):
    cytr[c,:].sort()

  return cytr
  
cytrda_b = comp_cyboot(ytr_da_b)
cytrda95 = [np.average(cytrda_b[:,24],weights=plA),np.average(cytrda_b[:,974],weights=plA)]
cytrwa_b = comp_cyboot(ytr_wa_b) 
cytrwa95 = [np.average(cytrwa_b[:,24],weights=plA),np.average(cytrwa_b[:,974],weights=plA)]

tr_comb95 = [[tres.loc[25].values[0],tres.loc[975].values[0]],cytrda95,cytrwa95]

##### Adaptation to Climate Change estimate 
# no area averaging

# at the county level
tmpidx = pd.DataFrame(yd_mat)
tmpidx.index = [yd_mat.index.get_level_values(0),yd_mat.index.get_level_values(1)+1981]
idx = tmpidx.index
colbeta =  pd.DataFrame([beta.values]*yd_mat.shape[0],index=tmpidx.index,columns=cols)
cy_w80 = (d_w80.loc[idx,cols] - d_clim.loc[idx,cols])*colbeta
cy_wld = (d_wld.loc[idx,cols] - d_clim.loc[idx,cols])*colbeta
c_durdiff = cy_wld.sum(axis=1) - cy_w80.sum(axis=1)
cy_w80_5 = (d_w80_5.loc[idx,cols] - d_clim.loc[idx,cols])*colbeta
cy_wld_5 = (d_wld_5.loc[idx,cols] - d_clim.loc[idx,cols])*colbeta
c_durdiff_5 = cy_wld_5.sum(axis=1) - cy_w80_5.sum(axis=1)
cy_w80_15 = (d_w80_15.loc[idx,cols] - d_clim.loc[idx,cols])*colbeta
cy_wld_15 = (d_wld_15.loc[idx,cols] - d_clim.loc[idx,cols])*colbeta
c_durdiff_15 = cy_wld_15.sum(axis=1) - cy_w80_15.sum(axis=1)

boot_mcdd = list()

for i in range(1000):
  ridx = np.random.randint(0,27115,27115)
  boot_mcdd.append(c_durdiff.iloc[ridx].mean())

# one sided test
np.where(boot_mcdd<0)

# get R^2 from decomposition
d_devy = d_deva.loc[idx,cols] - d_clim.loc[idx,cols]
d_weay = d_weaa.loc[idx,cols] - d_clim.loc[idx,cols]
d_dcomp = d_devy + d_weay
d_dcomp['Year'] = d_dcomp.index.get_level_values(1)-1981
d_dcomp['geoid'] = map(str,d_dcomp.index.get_level_values(0))

out = mod.predict(d_dcomp)
dr2 = list()

for c in ct:
  dr2.append(np.corrcoef(out.loc[c],yd_mat.loc[c].Yld)[0,1]**2)

# aggregate to region without area average
m_dd = c_durdiff.groupby(c_durdiff.index.get_level_values(1)).mean()
m_dd_5 = c_durdiff_5.groupby(c_durdiff.index.get_level_values(1)).mean()
m_dd_15 = c_durdiff_15.groupby(c_durdiff.index.get_level_values(1)).mean()

# aggregate to year
x = m_dd.index
b = np.polyfit(x-1981,m_dd.values,1)
b5 = np.polyfit(x-1981,m_dd_5.values,1)
b15 = np.polyfit(x-1981,m_dd_15.values,1)

ix = np.in1d(x,[1983,1988,2012],invert=True)
bo = np.polyfit(x[ix]-1981,m_dd.values[ix],1)
bo5 = np.polyfit(x[ix]-1981,m_dd_5.values[ix],1)
bo15 = np.polyfit(x[ix]-1981,m_dd_15.values[ix],1)

# uncertainty
ddboot = list()
dd5boot = list()
dd15boot = list()
ddoboot = list()
ddo5boot = list()
ddo15boot = list()
trdd_yrs = np.zeros([36,1000])
trdd_oyrs = np.zeros([36,1000])

for i in range(1000):
  ridx = np.random.randint(0,35,36)
  xidx = x[ridx] - 1981
  bb = np.polyfit(xidx,m_dd.values[ridx],1)
  ddboot.append(bb[0])
  trdd_yrs[:,i] = np.array(range(36))*bb[0]+bb[1]
  dd5boot.append(np.polyfit(xidx,m_dd_5.values[ridx],1)[0]) 
  dd15boot.append(np.polyfit(xidx,m_dd_15.values[ridx],1)[0])
  roidx = np.random.randint(0,32,33)
  xo = x[ix][roidx]-1981
  bbo = np.polyfit(xo,m_dd.values[ix][roidx],1)
  ddoboot.append(bbo[0])
  trdd_oyrs[:,i] = np.array(range(36))*bbo[0]+bbo[1]
  ddo5boot.append(np.polyfit(xo,m_dd_5.values[ix][roidx],1)[0]) 
  ddo15boot.append(np.polyfit(xo,m_dd_15.values[ix][roidx],1)[0])

## bootstrap p-value estimates for values <0
# len(np.where(np.array(ddboot)<0)[0])/1000.0

## 95% CIs
trdds = np.sort(trdd_yrs)
trdd95 = [trdds[:,24],trdds[:,974]]
trddos = np.sort(trdd_oyrs)
trddo95 = [trddos[:,24],trddos[:,974]]

## trends in spring temperature
# get final counties
geo = np.unique(yd_mat.geoid)
# tmin
ci = pd.read_csv('../weather_data/cou_tmi.csv',index_col='Date')
ci = pd.DataFrame(ci.values,index=pd.to_datetime(ci.index),columns=ci.columns)
ci = ci.loc[:,geo]
# tmax
cx = pd.read_csv('../weather_data/cou_tmx.csv',index_col='Date')
cx = pd.DataFrame(cx.values,index=pd.to_datetime(cx.index),columns=cx.columns)
cx = cx.loc[:,geo]

g = ci.groupby(pd.TimeGrouper(freq='M'))
cim = g.mean()

cimA = list()
for i in range(cim.shape[0]):
  cimA.append(np.average(cim.iloc[i,:],weights=plA))

cimA = pd.DataFrame(cimA,index=cim.index)

apr_may_ci = (cimA[cimA.index.month==4].values+cimA[cimA.index.month==5].values)/2
am_tr_ci = np.polyfit(range(36),apr_may_ci.ravel(),1)

g = cx.groupby(pd.TimeGrouper(freq='M'))
cxm = g.mean()

cxmA = list()
for i in range(cxm.shape[0]):
  cxmA.append(np.average(cxm.iloc[i,:],weights=plA))

cxmA = pd.DataFrame(cxmA,index=cxm.index)

apr_may_cx = (cxmA[cxmA.index.month==4].values+cxmA[cxmA.index.month==5].values)/2
am_tr_cx = np.polyfit(range(36),apr_may_cx.ravel(),1)




# calculate trend with and without drought years at county level
#x = c_durdiff.index.get_level_values(1)
#b = np.polyfit(x-1981,c_durdiff.values,1)
#b5 = np.polyfit(x-1981,c_durdiff_5.values,1) 
#b15 = np.polyfit(x-1981,c_durdiff_15.values,1)
#
#ix = np.in1d(x,[1983,1988,2012],invert=True)
#bo = np.polyfit(x[ix]-1981,c_durdiff.values[ix],1)
#bo5 = np.polyfit(x[ix]-1981,c_durdiff_5.values[ix],1)
#bo15 = np.polyfit(x[ix]-1981,c_durdiff_15.values[ix],1)
#
## uncertainty
#ddboot = list()
#dd5boot = list()
#dd15boot = list()
#ddoboot = list()
#ddo5boot = list()
#ddo15boot = list()
#trdd_yrs = np.zeros([36,1000])
#trdd_oyrs = np.zeros([36,1000])
#
#for i in range(1000):
#  ridx = np.random.randint(0,27115,27115)
#  xidx = x[ridx] - 1981
#  bb = np.polyfit(xidx,c_durdiff.values[ridx],1)
#  ddboot.append(bb[0])
#  trdd_yrs[:,i] = np.array(range(36))*bb[0]+bb[1]
#  dd5boot.append(np.polyfit(xidx,c_durdiff_5.values[ridx],1)[0]) 
#  dd15boot.append(np.polyfit(xidx,c_durdiff_15.values[ridx],1)[0])
#  roidx = np.random.randint(0,24845,24845)
#  xo = x[ix][roidx]-1981
#  bbo = np.polyfit(xo,c_durdiff.values[ix][roidx],1)
#  ddoboot.append(bbo[0])
#  trdd_oyrs[:,i] = np.array(range(36))*bbo[0]+bbo[1]
#  ddo5boot.append(np.polyfit(xo,c_durdiff_5.values[ix][roidx],1)[0]) 
#  ddo15boot.append(np.polyfit(xo,c_durdiff_15.values[ix][roidx],1)[0])
#
#trdds = np.sort(trdd_yrs)
#trdd95 = [trdds[:,24],trdds[:,974]]
#trddos = np.sort(trdd_oyrs)
#trddo95 = [trddos[:,24],trddos[:,974]]

## error from disaggregation
#arcomp = ar_gk_clim + (ar_gk_deva - ar_gk_clim) + (ar_gk_weaa - ar_gk_clim)
#se = (arcomp - ar_gk)**2
#rmse = np.sqrt(se.mean())

## correlation
#[np.corrcoef(arcomp.iloc[:,i],ar_gk.iloc[:,i])[0,1] for i in range(6)]

## Area weighted means? 

# uncertainty on trend incorporating both uncertainty in beta and included years
#c_dd = d_wld.loc[idx,cols] - d_w80.loc[idx,cols]
#c_dd_5 = d_wld_5.loc[idx,cols] - d_w80_5.loc[idx,cols]
#c_dd_15 = d_wld_5.loc[idx,cols] - d_w80_15.loc[idx,cols]
#
## aggregate to region
#pl.index = [pl.index.get_level_values(0),pl.Year]
#ar_dd = ar_summary(c_dd)
#ar_dd_5 = ar_summary(c_dd_5)
#ar_dd_15 = ar_summary(c_dd_15)
#
#ardd_yrs = np.zeros([36,1000])
#ardd_yrs_5 = np.zeros([36,1000])
#ardd_yrs_15 = np.zeros([36,1000])
#
#for i in range(1000):
#  bi = np.random.randint(1000)
#  br = betas.iloc[bi,:].values
#  tmpb = pd.DataFrame([br]*36,index=range(1981,2017),columns=cols)
#  ardd_yrs[:,i] = (ar_dd*tmpb).sum(axis=1)
#  ardd_yrs_5[:,i] = (ar_dd_5*tmpb).sum(axis=1)
#  ardd_yrs_15[:,i] = (ar_dd_15*tmpb).sum(axis=1)
#
## get trend estimates and expand to yearly values
#trdd_yrs = np.zeros([36,1000])
#dd_b = np.zeros([2,1000])
#dd_b_5 = np.zeros([2,1000])
#dd_b_15 = np.zeros([2,1000])
#
## omit high loss years 1983,1988,2012.
#yo12 = range(1981,2017)
##[yo12.remove(i) for i in [1983,1988,2012]]
#ix = np.in1d(yo12,[1983,1988,2012],invert=True)
#yo12 = np.array(yo12)[ix]-1981
#trdd_oyrs = np.zeros([36,1000])
#dd_ob = np.zeros([2,1000])
#dd_ob_5 = np.zeros([2,1000])
#dd_ob_15 = np.zeros([2,1000])
#
#for i in range(1000):
#  yi = np.random.randint(0,35,36) 
#  b = np.polyfit(yi,ardd_yrs[yi,i],1)
#  dd_b[:,i] = b
#  trdd_yrs[:,i] = np.array(range(36))*b[0]+b[1]
#  b = np.polyfit(yi,ardd_yrs_5[yi,i],1)
#  dd_b_5[:,i] = b
#  b = np.polyfit(yi,ardd_yrs_15[yi,i],1)
#  dd_b_15[:,i] = b
#  # omit 1983, 1988, 1993, and 2012
#  yoi = np.random.randint(0,31,32)
#  b = np.polyfit(yo12[yoi],ardd_yrs[yo12[yoi],i],1)
#  dd_ob[:,i] = b
#  trdd_oyrs[:,i] = np.array(range(36))*b[0]+b[1]
#  b = np.polyfit(yo12[yoi],ardd_yrs_5[yo12[yoi],i],1)
#  dd_ob_5[:,i] = b
#  b = np.polyfit(yo12[yoi],ardd_yrs_15[yo12[yoi],i],1)
#  dd_ob_15[:,i] = b
#
#dds = np.sort(dd_b)
#ddos = np.sort(dd_ob)
#trdds = np.sort(trdd_yrs)
#trdd95 = [trdds[:,24],trdds[:,974]]
#trddos = np.sort(trdd_oyrs)
#trddo95 = [trddos[:,24],trddos[:,974]]
#
#dds_5 = np.sort(dd_b_5)
#ddos_5 = np.sort(dd_ob_5)
#dds_15 = np.sort(dd_b_15)
#ddos_15 = np.sort(dd_ob_15)
#
## bootstrap p-value estimates for values <0
#pddos = len(np.where(ddos[0,:]<0)[0])/float(ddos.shape[1])

# extraneous?
#c_durdiffA = pd.DataFrame(c_durdiff)
#c_durdiffA = c_durdiffA.join(pl.Area)
#c_durdiffA.columns = ['dd','Area']
