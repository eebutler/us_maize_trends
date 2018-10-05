import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from sklearn import cluster

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
col[col=='Year.1'] = 'Year'
col[col=='geoid.1'] = 'geoid'
yd_mat.columns = col
# convert geoid to string to access both state and county components
yd_mat.geoid = map(str,yd_mat.geoid)

# weather DFs
d_mat = pd.read_csv('d_mat.csv',index_col=[0,1])
#d_mat = d_mat.iloc[:,0:9]
col = d_mat.columns.values
col[0] = 'Year'
d_mat.columns = col
d_clim = pd.read_csv('d_clim.csv',index_col=[0,1])
d_clim.columns = col
d_dev = pd.read_csv('d_dev.csv',index_col=[0,1])
d_dev.columns = col
d_wea = pd.read_csv('d_wea.csv',index_col=[0,1])
d_wea.columns = col
d_gkpmeans = pd.read_csv('d_gkpmeans.csv',index_col=[0,1])
d_gkpmeans.columns = col[0:13] # will aggregate to seasonal totals later
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
# calculate with precipitation?
prec = False

# state names
stnms = np.unique(stp.index.get_level_values(0))

# State ID numbers 
fips = map(str,[17,18,19,20,21,26,27,29,31,39,46,55])

# State Names
stn = ['ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY','MICHIGAN',
       'MINNESOTA','MISSOURI','NEBRASKA','OHIO','SOUTH DAKOTA','WISCONSIN']

# cross validate
idx = map(int,np.linspace(0,yd_mat.shape[0]-1,yd_mat.shape[0]))
samp = np.random.choice(idx,int(np.round(len(idx)*0.8)),replace=False)
samp = np.sort(samp)
osamp = np.setdiff1d(idx,samp)
yd_mat_sub = yd_mat.iloc[samp,:]
yd_mat_test = yd_mat.iloc[osamp,:]

# extract columns
acols = list(yd_mat.columns)
# omit drydown only G/K + season P and P^2
fcols = acols[0:4] + acols[5:8] 
if prec:
  fcols = acols[0:4] + acols[5:8] + acols[13:15]  
form = 'Yld ~ C(geoid)+'+'+'.join(fcols)
mod = smf.ols(formula=form,data=yd_mat).fit()
sub_mod = smf.ols(formula=form,data=yd_mat_sub).fit()
test_pred = sub_mod.predict(yd_mat_test)
rsq_test = np.corrcoef(test_pred.values,yd_mat_test.Yld.values)[0,1]**2

# get county level R^2
cr2 = []
fva = mod.fittedvalues

ra = len(fcols)-1
beta = mod.params[-ra:]
tr = mod.params[-(ra+1)]

# individual county fits
ct = yd_mat.index.get_level_values(0).unique()
for c in ct:
  r2 = np.corrcoef(fva.loc[c],yd_mat.loc[c].Yld)[0,1]**2  
  cr2.append(r2)

cr2 = pd.DataFrame(cr2,index=ct)
#cr2.to_csv('cr2.csv')

# group by lat/lon and mean yield to account for spatial autocorrelation
clus = yd_mat.loc[:,['Yld','lat','lon']].groupby('geoid').mean()
# need to specify random state for consistent results
nclus = 108
km108 = cluster.KMeans(n_clusters=nclus,random_state=42)
km108cls = km108.fit(clus.loc[:,['Yld','lat','lon']].values)
clus108 = pd.DataFrame(km108cls.labels_,index=clus.index,columns=['cluster'])
yd_clus = yd_mat.join(clus108)
yd_clus.index = [yd_clus.cluster,yd_clus.Year]
yd_clus = yd_clus.sort_index(level=0)

# bootstrap on clusters
## storage list

beta_boot = np.zeros([ra,1000])
tre_boot = np.zeros(1000)

if boot_ests:
  i = 0
  while i < 1000:
    try:
      bcl = []                                                      
      for j in range(nclus):
        rdcl = np.random.randint(nclus)
        tmp = yd_clus.loc[rdcl]
        ind = np.random.randint(tmp.shape[0],size=tmp.shape[0])
        # randomize years within cluster
        bcl.append(tmp.iloc[ind].values)
      bcl = np.vstack(bcl)
      bcl = pd.DataFrame(bcl,columns=yd_clus.columns)
      bcl = bcl.astype(float)
      #bcl[['geoid','cluster','Year']].astype(int) 
      out = smf.ols(formula=form,data=bcl).fit()
      beta_boot[:,i] = out.params[-ra:]
      tre_boot[i] = out.params[-(ra+1)]
      i += 1
      if i%50==0: print i
    except LinAlgError:
      continue
  # betas
  tmp = np.sort(beta_boot,axis=1)
  betas = pd.DataFrame(tmp.T,columns=fcols[1:ra+1])
  betas.index = range(1,1001)
  # 95% CI
  # trend
  tres = pd.DataFrame(np.sort(tre_boot),index=range(1,1001),columns=['trend'])
  betas.to_csv('beta_b.csv')
  tres.to_csv('trend_b.csv')
else:
  betas = pd.read_csv('beta_b.csv',index_col=0)
  tres = pd.read_csv('trend_b.csv',index_col=0)

# to get 95% CI
#beta95 = np.array(betas)
#beta95 = np.sort(beta95,axis=0)
#beta95[[24,974],:]

## get mean values Duration for each year and state
# remove Year from cols
#cols = cols[1:4] + cols[5:8]
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
             'Dry_KDD':lambda x: np.average(x.Dry_KDD, weights=x.Area),
	     'Veg_PRC':lambda x: np.average(x.Veg_PRC, weights=x.Area),   
             'EGF_PRC':lambda x: np.average(x.EGF_PRC, weights=x.Area),
             'LGF_PRC':lambda x: np.average(x.LGF_PRC, weights=x.Area),
             'Dry_PRC':lambda x: np.average(x.Dry_PRC, weights=x.Area)}

  # column indices for looping
  cols = yda_mat.columns[1:13]
  
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
st_means = st_summary(d_gkpmeans)

kd_sens = np.zeros([len(stnms),4])
kd_fit = np.zeros([len(stnms),4])
kd_sens_boot = np.zeros([len(stnms),4,1000])
for i,s in enumerate(stnms):
  sz = st_means.loc[s].shape[0]
  idx = np.random.randint(sz,size=[sz,1000])
  for j in range(4):
    Y = st_means.loc[s].iloc[:,12+j]
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
st_grm = st_means.groupby(level=0).mean()
#st_grm = st_gr.mean()

# dict to connect fips with state names
fsd = dict(zip(fips,stnms))

# all county geoids
ct = d_gkpmeans.index.get_level_values(0).unique()

# values to take anomalies
cols = acols[1:4]+acols[5:8]+acols[9:12]

gkp_veg_anoms = pd.DataFrame()
dys_veg_anoms = pd.DataFrame()
for c in ct:
  # get the state name from the geoid
  s = fsd[str(c)[0:2]]
  anoms = d_gkpmeans.loc[c,cols] - st_grm.loc[s,cols]
  y = d_gkpmeans.loc[c].shape[0]
  # day anomaly from yearly KDD anomaly and KDD-Day sensitivity
  dys = anoms.iloc[:,3:6].values*np.array([kd_sens.loc[s,['Veg','EGF','LGF']]]*y)
  # collect anomalies in each phase 
  tmp = np.tile(dys,3)*([np.array(st_grm.loc[s,cols])]*y)
  tmp = pd.DataFrame(tmp,columns=cols)
  gkp_veg_anoms = gkp_veg_anoms.append(tmp)
  dys = pd.DataFrame(dys,columns=['Veg','EGF','LGF'])
  dys_veg_anoms = dys_veg_anoms.append(dys)

gkp_veg_anoms.index = d_gkpmeans.index
dys_veg_anoms.index = d_gkpmeans.index

# subtract anomaly from dev and add to wea
d_deva = d_dev.loc[:,cols] - gkp_veg_anoms
d_deva['PRC'] = d_deva.iloc[:,6:9].sum(axis=1)
d_deva['PRC2'] = d_deva['PRC']**2
d_deva['Year'] = d_dev.Year

d_weaa = d_wea.loc[:,cols] + gkp_veg_anoms 
d_weaa['PRC'] = d_weaa.iloc[:,6:9].sum(axis=1)
d_weaa['PRC2'] = d_weaa['PRC']**2
d_weaa['Year'] = d_wea.Year
d_w80a = d_w80.loc[:,cols] + gkp_veg_anoms 
d_w80a['Year'] = d_w80.Year
d_wlda = d_wld.loc[:,cols] + gkp_veg_anoms 
d_wlda['Year'] = d_wld.Year

# unique county geoids with yields
ct = yd_mat.index.get_level_values(0).unique()

def ctrends(dmat,da_or_wa,cols):
  ctr = pd.DataFrame(columns=cols)
  ctr_b = np.zeros([len(ct),len(cols),1000]) 
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
	  # doesn't currently work with precip estimates...
          si = np.where(np.array(fips)==str(c)[0:2])[0][0]
          bi = np.random.randint(1000,size=3)
          b = [kd_sens_boot[si,r,bi[r]] for r in range(3)]
	  if prec:
	    acols = ['Veg_GDD','EGF_GDD','LGF_GDD',
		     'Veg_KDD','EGF_KDD','LGF_KDD',
		     'Veg_PRC','EGF_PRC','LGF_PRC'] 
	    anoms = d_gkpmeans.loc[c,acols] - st_grm.loc[s,acols]        	 
            y = d_gkpmeans.loc[c].shape[0]
            dys = anoms.iloc[:,3:6].values*np.tile(b,[y,1])
            ntile = len(st_grm.loc[s,acols])/3
            gka = np.tile(dys,ntile)*([np.array(st_grm.loc[s,acols])]*y)
	    Y = dmat.loc[c,acols].iloc[yd_mat.loc[c].Year.values]
	    Y.index = X.index
	    if da_or_wa == 'da':
	      Yi = Y.iloc[ridx] - gka[ridx]
	      Yi['PRC'] = Yi.iloc[:,6:9].sum(axis=1)
	      Yi['PRC2'] = Yi['PRC']**2
	    else:
	      Yi = Y.iloc[ridx] + gka[ridx]
	      Yi['PRC'] = Yi.iloc[:,6:9].sum(axis=1)	
              Yi['PRC2'] = Yi['PRC']**2
	  else: 
	    anoms = d_gkpmeans.loc[c,cols] - st_grm.loc[s,cols]  
            y = d_gkpmeans.loc[c].shape[0]
            dys = anoms.iloc[:,3:6].values*np.tile(b,[y,1])
	    ntile = len(st_grm.loc[s,cols])/3
            gka = np.tile(dys,ntile)*([np.array(st_grm.loc[s,cols])]*y)
	    if da_or_wa == 'da':
	      Yi = Y.iloc[ridx] - gka[ridx]
	    else:
	      Yi = Y.iloc[ridx] + gka[ridx]
        for j,p in enumerate(cols):
          b = sm.OLS(Yi.loc[:,p],Xi).fit().params
          ctr_b[k,j,i] = b[1]

  return [ctr,ctr_b]

ctr,ctr_b = ctrends(d_mat,False,fcols[1:])
ctr_d,ctr_d_b = ctrends(d_dev,False,fcols[1:])
ctr_w,ctr_w_b = ctrends(d_wea,False,fcols[1:])
ctr_da,ctr_da_b = ctrends(d_deva,'da',fcols[1:])  
ctr_wa,ctr_wa_b = ctrends(d_weaa,'wa',fcols[1:])
# only run these interactively to get plant plasticity contributions
#gkpatr,gkpatr_b = ctrends(gkp_veg_anoms,False,fcols[1:])
#datr = ctrends(dys_veg_anoms,False,['Veg','EGF','LGF'])

#ctr.to_csv('ctr.csv')
#ctr_d.to_csv('ctr_d.csv')
#ctr_w.to_csv('ctr_w.csv')
#ctr_da.to_csv('ctr_da.csv')
#ctr_wa.to_csv('ctr_wa.csv')

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
  y = len(fcols)-1
  ctr_b = np.loadtxt('ctr_b.csv',delimiter=',').reshape(cts,y,1000)
  ctr_d_b = np.loadtxt('ctr_d_b.csv',delimiter=',').reshape(cts,y,1000)
  ctr_w_b = np.loadtxt('ctr_w_b.csv',delimiter=',').reshape(cts,y,1000)
  ctr_da_b = np.loadtxt('ctr_da_b.csv',delimiter=',').reshape(cts,y,1000)
  ctr_wa_b = np.loadtxt('ctr_wa_b.csv',delimiter=',').reshape(cts,y,1000)

# expand beta estimates to each county number of counties in each state
cb = np.tile(beta,[ctr.shape[0],1])
cbetas = pd.DataFrame(cb,columns=fcols[1:9],index=ctr.index)
#cbetas.to_csv('cbetas.csv')
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

# iterator for individual predictors
ra = len(fcols)-1
# mean yield trend from development shifts
ytmd = np.average(y_trd.sum(axis=1),weights=plA)
ytdp = ytmd/tr

ytmdf = [np.average(y_trd.iloc[:,i],weights=plA) for i in range(ra)]
ytmdfp = ytmdf/tr

# with plant plasticity
ytmda = np.average(y_trda.sum(axis=1),weights=plA)
ytdap = ytmda/tr

ytmdaf = [np.average(y_trda.iloc[:,i],weights=plA) for i in range(ra)]
ytmdafp = ytmdaf/tr

# mean yield trend from climate
ytmw = np.average(y_trw.sum(axis=1),weights=plA)
ytwp = ytmw/tr

ytmwf = [np.average(y_trw.iloc[:,i],weights=plA) for i in range(ra)]
ytmwfp = ytmwf/tr

# with plant plasticity
ytmwa = np.average(y_trwa.sum(axis=1),weights=plA)
ytwap = ytmwa/tr

ytmwaf = [np.average(y_trwa.iloc[:,i],weights=plA) for i in range(ra)]
ytmwafp = ytmwaf/tr

# mean yield trend without separating effects
ytm = np.average(y_tr.sum(axis=1),weights=plA)
ytp = ytm/tr

ytmf = [np.average(y_tr.iloc[:,i],weights=plA) for i in range(ra)]
ytmfp = ytmf/tr

# planting area weighted
# bootstrap gdd/kdd and yield trends estimates

trpA = [np.average(ctr.iloc[:,i],weights=plA) for i in range(ra)]
trdpA = [np.average(ctr_d.iloc[:,i],weights=plA) for i in range(ra)]
trwpA = [np.average(ctr_w.iloc[:,i],weights=plA) for i in range(ra)]
trdapA = [np.average(ctr_da.iloc[:,i],weights=plA) for i in range(ra)]
trwapA = [np.average(ctr_wa.iloc[:,i],weights=plA) for i in range(ra)]

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
          np.average(tr_b[:,i,974],weights=plA)] for i in range(ra)]
trd95 = [[np.average(tr_d_b[:,i,24],weights=plA),
          np.average(tr_d_b[:,i,974],weights=plA)] for i in range(ra)]
trw95 = [[np.average(tr_w_b[:,i,24],weights=plA),
          np.average(tr_w_b[:,i,974],weights=plA)] for i in range(ra)]
trda95 = [[np.average(tr_da_b[:,i,24],weights=plA),
           np.average(tr_da_b[:,i,974],weights=plA)] for i in range(ra)]
trwa95 = [[np.average(tr_wa_b[:,i,24],weights=plA),
	    np.average(tr_wa_b[:,i,974],weights=plA)] for i in range(ra)]

# convert into yield trends
# assumes if using subset of predictors that
def cyboot(ctr_nd,cols):
  ytr = np.zeros([len(ct),cols,1000]) 
  for i in range(1000):
    ridx = np.random.randint(1000)
    ridxb = np.random.randint(1000)
    for c in range(len(ct)):
      ytr[c,:,i] = ctr_nd[c,0:cols,ridx]*betas.iloc[ridxb,:].values
  for c in range(len(ct)):
    ytr[c,:,:].sort(axis=1)

  return ytr

ytr_b = cyboot(ctr_b,len(fcols[1:]))
ytr_d_b = cyboot(ctr_d_b,len(fcols[1:]))
ytr_w_b = cyboot(ctr_w_b,len(fcols[1:]))
ytr_da_b = cyboot(ctr_da_b,len(fcols[1:]))
ytr_wa_b = cyboot(ctr_wa_b,len(fcols[1:]))

ytr95 = [[np.average(ytr_b[:,i,24],weights=plA),
	   np.average(ytr_b[:,i,974],weights=plA)] for i in range(ra)]
ytrd95 = [[np.average(ytr_d_b[:,i,24],weights=plA),
	   np.average(ytr_d_b[:,i,974],weights=plA)] for i in range(ra)]
ytrw95 = [[np.average(ytr_w_b[:,i,24],weights=plA),
	   np.average(ytr_w_b[:,i,974],weights=plA)] for i in range(ra)]
ytrda95 = [[np.average(ytr_da_b[:,i,24],weights=plA),
	    np.average(ytr_da_b[:,i,974],weights=plA)] for i in range(ra)]
ytrwa95 = [[np.average(ytr_wa_b[:,i,24],weights=plA),
	    np.average(ytr_wa_b[:,i,974],weights=plA)] for i in range(ra)]

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

# avg plant plasticity duration change across region
# datr_m = np.average(datr.sum(axis=1),weights=plA)

# area average yield
pl.index = [pl.index.get_level_values(0),pl.Year-1981]
tmp = yd_mat.join(pl.Area)
g = tmp.groupby(tmp.Year)
yavg = g.apply(lambda x: np.average(x.Yld,weights=x.Area))
yavg.index = yavg.index+1981
pl.index = [pl.index.get_level_values(0),pl.Year]

def ar_summary(dmat,cols):
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
             'LGF_KDD':lambda x: np.average(x.LGF_KDD, weights=x.Area),
	     'PRC':lambda x: np.average(x.PRC, weights=x.Area),
	     'PRC2':lambda x: np.average(x.PRC2, weights=x.Area)}

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

ar_gkp = ar_summary(d_mat,fcols[1:])
ar_gkp_clim = ar_summary(d_clim,fcols[1:])
#ar_gk_dev = ar_summary(d_dev,fcols[1:])
#ar_gk_wea = ar_summary(d_wea,fcols[1:])
ar_gkp_deva = ar_summary(d_deva,fcols[1:])
ar_gkp_weaa = ar_summary(d_weaa,fcols[1:])
ar_gkp_w80 = ar_summary(d_w80,fcols[1:])
ar_gkp_wld = ar_summary(d_wld,fcols[1:])
ar_gkp_w80_5 = ar_summary(d_w80_5,fcols[1:])
ar_gkp_wld_5 = ar_summary(d_wld_5,fcols[1:])
ar_gkp_w80_15 = ar_summary(d_w80_15,fcols[1:])
ar_gkp_wld_15 = ar_summary(d_wld_15,fcols[1:])

# expand beta values for whole region average
ybeta = pd.DataFrame([beta.values]*37,index=range(1981,2018),columns=fcols[1:9])

# convert to yield
yld_pcomp = (ar_gkp - ar_gkp_clim)*ybeta
yld_compda = (ar_gkp_deva - ar_gkp_clim)*ybeta
yld_compwa = (ar_gkp_weaa - ar_gkp_clim)*ybeta
yld_compw80 = (ar_gkp_w80 - ar_gkp_clim)*ybeta
yld_compwld = (ar_gkp_wld - ar_gkp_clim)*ybeta
yld_compw80_5 = (ar_gkp_w80_5 - ar_gkp_clim)*ybeta
yld_compwld_5 = (ar_gkp_wld_5 - ar_gkp_clim)*ybeta
yld_compw80_15 = (ar_gkp_w80_15 - ar_gkp_clim)*ybeta
yld_compwld_15 = (ar_gkp_wld_15 - ar_gkp_clim)*ybeta

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
if prec:
  ydaP = yld_compda.loc[:,['PRC','PRC2']].sum(axis=1)
  ywaP = yld_compwa.loc[:,['PRC','PRC2']].sum(axis=1)

if prec:
  yld_comb = pd.DataFrame([yavg,ydaG,ydaK,ydaP,ywaG,ywaK,ywaP]).T
  yld_comb.columns = ['yld','Dev_GDD','Dev_KDD','Dev_PRC','Wea_GDD','Wea_KDD','Wea_PRC']
else:
  yld_comb = pd.DataFrame([yavg,ydaG,ydaK,ywaG,ywaK]).T 
  yld_comb.columns = ['yld','Dev_GDD','Dev_KDD','Wea_GDD','Wea_KDD']

#yld_comb.to_csv('yld_comb.csv')

# best way to report trends?
tr_tmp = list()
for c in yld_comb.columns:
 tr_tmp.append(np.polyfit(yld_comb.index.values,yld_comb.loc[:,c],1)[0])

# trend components
tr_all = [tr,np.sum(ytmwaf[0:3]),np.sum(ytmwaf[3:6]),np.sum(ytmdaf[0:3]),np.sum(ytmdaf[3:6])]

tr_comb = [tr,np.sum(ytmwaf),np.sum(ytmdaf)]

if prec:
  tr_all = [tr,np.sum(ytmwaf[0:3]),np.sum(ytmwaf[3:6]),np.sum(ytmwaf[6:8]),
	       np.sum(ytmdaf[0:3]),np.sum(ytmdaf[3:6]),np.sum(ytmdaf[6:8])]

  tr_comb = [tr,np.sum(ytmwaf),np.sum(ytmdaf)]

# compound trend uncertainty
def comp_cyboot(ctr_nd):
  ytr = np.zeros([len(ct),ra,1000])
  cytr = np.zeros([len(ct),1000])
  for i in range(1000): 
    for c in range(len(ct)):
      for p in range(ra):
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

## check decomposition into development and weather
tmpidx = yd_mat.copy()
tmpidx.index = [tmpidx.index.get_level_values(0),tmpidx.index.get_level_values(1)+1981]
idx = tmpidx.index
cols = fcols[1:ra+1]

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

##### Adaptation to Climate Change estimate 

# county level yield differences
tmpidx = yd_mat.copy()
tmpidx.index = [tmpidx.index.get_level_values(0),tmp.index.get_level_values(1)+1981]
idx = tmpidx.index
cols = fcols[1:ra+1]
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

### decadal yield difference of means
boot_mcdd = list()

for i in range(1000):
  ridx = np.random.randint(0,c_durdiff.shape[0],c_durdiff.shape[0])
  boot_mcdd.append(c_durdiff.iloc[ridx].mean())

# one sided test
np.where(boot_mcdd<0)

dddf = pd.DataFrame(c_durdiff)
dddf = dddf.join(clus108['cluster'])
ddcl = dddf.groupby(['cluster','Year']).mean()
dddf5 = pd.DataFrame(c_durdiff_5)
dddf5 = dddf5.join(clus108['cluster'])
ddcl5 = dddf5.groupby(['cluster','Year']).mean()
dddf15 = pd.DataFrame(c_durdiff_15)
dddf15 = dddf15.join(clus108['cluster'])
ddcl15 = dddf15.groupby(['cluster','Year']).mean()

## bootstrap uncertainty
ddboot = list()
dd5boot = list()
dd15boot = list()
trdd_yrs = np.zeros([37,1000])

yrs = ddcl.index.get_level_values('Year')
for i in range(1000):
  ridx = np.random.randint(0,ddcl.shape[0],ddcl.shape[0])
  bb = np.polyfit(yrs[ridx],ddcl.values[ridx],1)
  ddboot.append(bb[0])
  trdd_yrs[:,i] = np.array(range(1981,2018))*bb[0]+bb[1]
  bb = np.polyfit(yrs[ridx],ddcl5.values[ridx],1) 
  dd5boot.append(bb[0])
  bb = np.polyfit(yrs[ridx],ddcl15.values[ridx],1) 
  dd15boot.append(bb[0])

## bootstrap p-value estimates for values <0
b = np.polyfit(ddcl.index.get_level_values(1),ddcl.values,1)
ddboot.sort()
ddboot[24],ddboot[974]
len(np.where(np.array(ddboot)<0)[0])/1000.0

## 5 year 
b5 = np.polyfit(ddcl5.index.get_level_values(1),ddcl5.values,1)
dd5boot.sort()
dd5_95 = np.array([dd5boot[24],dd5boot[974]])
len(np.where(np.array(dd5boot)<0)[0])/1000.0

## 15 year 
b15 = np.polyfit(ddcl15.index.get_level_values(1),ddcl15.values,1)
dd15boot.sort()
dd15_95 = np.array([dd15boot[24],dd15boot[974]])
len(np.where(np.array(dd15boot)<0)[0])/1000.0

## 95% CIs on each year
trdds = np.sort(trdd_yrs)
trdd95 = [trdds[:,24],trdds[:,974]]

## Different Spatial Clusters


## county 95% CI and p-value
dddf = pd.DataFrame(c_durdiff)
ddcboot = list()

yrs = dddf.index.get_level_values('Year')
for i in range(10000):
  ridx = np.random.randint(0,dddf.shape[0],dddf.shape[0])  
  bb = np.polyfit(yrs[ridx],dddf.values[ridx],1)
  ddcboot.append(bb[0])

bc = np.polyfit(dddf.index.get_level_values(1),dddf.values,1)
ddcboot.sort()
ddcboot[249],ddcboot[9749]
len(np.where(np.array(ddcboot)<0)[0])/10000.0


# regional 95% CI and p-value
ddr = dddf.groupby(['Year']).mean()
ddrboot = list()

yrs = ddr.index.get_level_values('Year')
for i in range(1000):
  ridx = np.random.randint(0,ddr.shape[0],ddr.shape[0])
  bb = np.polyfit(yrs[ridx],ddr.values[ridx],1)
  ddrboot.append(bb[0])

br = np.polyfit(ddr.index.get_level_values(0),ddr.values,1)
ddrboot.sort()
ddrboot[249],ddrboot[9749]
len(np.where(np.array(ddrboot)<0)[0])/10000.0

# regional 95% CI and p-value, w/o 2017
ddr2 = ddr.iloc[0:36]
ddr2boot = list()

yrs = ddr2.index.get_level_values('Year')
for i in range(1000):
  ridx = np.random.randint(0,ddr2.shape[0],ddr2.shape[0])
  bb = np.polyfit(yrs[ridx],ddr2.values[ridx],1)
  ddr2boot.append(bb[0])

br2 = np.polyfit(ddr2.index.get_level_values(0),ddr2.values,1)
ddr2boot.sort()
ddr2boot[24],ddr2boot[974]
len(np.where(np.array(ddr2boot)<0)[0])/1000.0

# regional planting area average

dddfA = dddf.join(pl.Area)
dddfA.columns = ['Ydiff','Area']
wm = lambda x: np.average(x,weights=dddfA.loc[x.index,"Area"])
ydwm = dddfA.groupby('Year').agg(wm)

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
am_tr_ci = np.polyfit(range(37),apr_may_ci.ravel(),1)

g = cx.groupby(pd.TimeGrouper(freq='M'))
cxm = g.mean()

cxmA = list()
for i in range(cxm.shape[0]):
  cxmA.append(np.average(cxm.iloc[i,:],weights=plA))

cxmA = pd.DataFrame(cxmA,index=cxm.index)

apr_may_cx = (cxmA[cxmA.index.month==4].values+cxmA[cxmA.index.month==5].values)/2
am_tr_cx = np.polyfit(range(37),apr_may_cx.ravel(),1)
