import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpat
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
#from statsmodels.regression.quantile_regression import QuantReg

yd_mat = pd.read_csv('yd_mat.csv',index_col=[0,1])
# straighten out the column names
col = yd_mat.columns.values
col[0] = 'Year'
col[col=='geoid.1'] = 'geoid'
yd_mat.columns = col
# convert geoid to string to access both state and county components
yd_mat.geoid = map(str,yd_mat.geoid)

geo = np.unique(yd_mat.geoid)

# state progress
st = pd.read_csv('../crop_data/progress.csv',index_col=[0,1])

st.index.names = ['State','Date']

stp = pd.DataFrame(data={'Veg':st.Planted-st.Silking,'EGF':st.Silking-st.Dough, 
			 'LGF':st.Dough-st.Mature, 'Dry':st.Mature-st.Harvest},
		   columns=['Veg','EGF','LGF','Dry'])

# remove NaN values using EGF column as index - Georgia started reporting
# silking and mature in 2016...
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

# tmin
ci = pd.read_csv('../weather_data/cou_tmi.csv',index_col='Date')
ci = pd.DataFrame(ci.values,index=pd.to_datetime(ci.index),columns=ci.columns)
ci = ci.loc[:,geo]
# tmax
cx = pd.read_csv('../weather_data/cou_tmx.csv',index_col='Date')
cx = pd.DataFrame(cx.values,index=pd.to_datetime(cx.index),columns=cx.columns)
cx = cx.loc[:,geo]

# geoid and year iterators
geoid = list(ci.columns)
yrs = map(str,range(1981,2017))

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

# state columns
tmp = map(str,cx.columns)
st = [t[0:2] for t in tmp] 

# State Mean Development for First and Last decade of record
# Iowa is 2
#for i in [2]: 
for i in range(len(stn)):
  fdg = stp.loc[stn[i],:].loc['1981':'1990']
  fdg = fdg.groupby(stp.loc[stn[i],:].loc['1981':'1990'].index.dayofyear)
  fdm = fdg.mean()
  
  ldg = stp.loc[stn[i],:].loc['2008':'2017']
  ldg = ldg.groupby(stp.loc[stn[i],:].loc['2008':'2017'].index.dayofyear)
  ldm = ldg.mean()
  
  ia = [s==fips[i] for s in st]
  
  kdd_fdg = kdd.loc['1981':'1990',ia].groupby(kdd.loc['1981':'1990',ia].index.dayofyear)
  kdd_fdm = kdd_fdg.mean().mean(axis=1)
  kdd_fdm = kdd_fdm.rolling(30).mean()
  
  kdd_ldg = kdd.loc['2008':'2017',ia].groupby(kdd.loc['2008':'2017',ia].index.dayofyear)
  kdd_ldm = kdd_ldg.mean().mean(axis=1)
  kdd_ldm = kdd_ldm.rolling(30).mean()
  
  gdd_fdg = gdd.loc['1981':'1990',ia].groupby(gdd.loc['1981':'1990',ia].index.dayofyear)
  gdd_fdm = gdd_fdg.mean().mean(axis=1)
  gdd_fdm = gdd_fdm.rolling(30).mean()
  
  gdd_ldg = gdd.loc['2008':'2017',ia].groupby(gdd.loc['2008':'2017',ia].index.dayofyear)
  gdd_ldm = gdd_ldg.mean().mean(axis=1)
  gdd_ldm = gdd_ldm.rolling(30).mean()
  
  days = np.linspace(1,366,366)
  
  # limit range
  # first day > 0 planting
  fd = np.where(ldm.Veg>0)[0][0]
  # last day > 0 late grain filling
  ld = np.where(ldm.LGF>0)[0][-1]
  rng = range(fd-15,ld+15)
  
  fig, host = plt.subplots()
  par = host.twinx()
  
  p1, = host.plot(days[rng],kdd_fdm[rng]+8,color=(1,0,0),label='fd_max')
  p3, = host.plot(days[rng],kdd_ldm[rng]+8,color=(0.5,0,0),label='ld_max')
  
  p2, = host.plot(days[rng],gdd_fdm[rng],color=(1,0.75,0.4),label='fd_gdd')
  p5, = host.plot(days[rng],gdd_ldm[rng],color=(1,0.5,0),label='ld_gdd')
  
  p4, = par.plot(days,fdm.Veg,color=(0,1,0),label='fd_veg')
  p6, = par.plot(days,ldm.Veg,color=(0,0.5,0),label='ld_veg')
  
  p7, = par.plot(days,fdm.EGF,color=(0,0,1),label='fd_egf')
  p9, = par.plot(days,ldm.EGF,color=(0,0,0.5),label='ld_egf')
  
  p10, = par.plot(days,fdm.LGF,color=(0,1,1),label='fd_lgf')
  p12, = par.plot(days,ldm.LGF,color=(0,0.5,0.5),label='ld_lgf')
  
  host.set_xlim(75,325)
  host.set_ylim(0,16)
  par.set_ylim(0,250)
  host.set_yticks(range(2,18,2))
  host.set_yticklabels(['2','4','6','8(0)','10(2)','12(4)','14(6)','16(8)'])
  par.set_yticks([0,50,100])
  par.set_yticklabels(['0','50','100'])
  
  
  host.set_xlabel('Day of Year')
  host.set_ylabel('Degree Days [$^\circ$C day]')
  
  par.text(347,100,'Development Phase [%]',rotation='vertical')
  
  lines = [p1,p3,p4,p6,p7,p9,p10,p12,p2,p5]
  
  t1, = plt.plot([],[],color=(1,0,0),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
  t2, = plt.plot([],[],color=(0.5,0,0),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')
  
  t3, = plt.plot([],[],color=(1,0.75,0.4),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
  t4, = plt.plot([],[],color=(1,0.5,0),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')
  
  v1, = plt.plot([],[],color=(0,1,0),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
  v2, = plt.plot([],[],color=(0,0.5,0),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')
  
  e1, = plt.plot([],[],color=(0,0,1),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
  e2, = plt.plot([],[],color=(0,0,0.5),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')
  
  l1, = plt.plot([],[],color=(0,1,1),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
  l2, = plt.plot([],[],color=(0,0.5,0.5),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')
  
  plt.legend(((t1, t2), (t3, t4), (v1, v2), (e1,e2), (l1,l2)), 
  	   ('KDD','GDD','Vegetative','Early Grain Fill','Late Grain Fill'),
              loc='upper left', bbox_to_anchor=(0.024,0.95), labelspacing=1.1, frameon=False)
  
  plt.text(78,237,'Early|Recent',fontsize=9)
  
  plt.savefig(stn[i].replace(' ','_')+'_kd_dev.pdf')
  plt.close()


### whole region
stp.index = stp.index.get_level_values(1)
fdg = stp.loc['1981':'1990']
fdg = fdg.groupby(fdg.index.dayofyear)
fdm = fdg.mean()

ldg = stp.loc['2007':'2016']
ldg = ldg.groupby(ldg.index.dayofyear)
ldm = ldg.mean()

kdd_fdg = kdd.loc['1981':'1990',:].groupby(kdd.loc['1981':'1990',:].index.dayofyear)
kdd_fdm = kdd_fdg.mean().mean(axis=1)
kdd_fdm = kdd_fdm.rolling(30).mean()

kdd_ldg = kdd.loc['2008':'2017',:].groupby(kdd.loc['2008':'2017',:].index.dayofyear)
kdd_ldm = kdd_ldg.mean().mean(axis=1)
kdd_ldm = kdd_ldm.rolling(30).mean()

gdd_fdg = gdd.loc['1981':'1990',:].groupby(gdd.loc['1981':'1990',:].index.dayofyear)
gdd_fdm = gdd_fdg.mean().mean(axis=1)
gdd_fdm = gdd_fdm.rolling(30).mean()

gdd_ldg = gdd.loc['2008':'2017',:].groupby(gdd.loc['2008':'2017',:].index.dayofyear)
gdd_ldm = gdd_ldg.mean().mean(axis=1)
gdd_ldm = gdd_ldm.rolling(30).mean()

days = np.linspace(1,366,366)

# limit range
# first day > 0 planting
fd = np.where(ldm.Veg>0)[0][0]
# last day > 0 late grain filling
ld = np.where(ldm.LGF>0)[0][-1]
rng = range(fd-15,ld+15)

fig, host = plt.subplots()
par = host.twinx()

p1, = host.plot(days[rng],kdd_fdm[rng]+8,color=(1,0,0),label='fd_max')
p3, = host.plot(days[rng],kdd_ldm[rng]+8,color=(0.5,0,0),label='ld_max')

p2, = host.plot(days[rng],gdd_fdm[rng],color=(1,0.75,0.4),label='fd_gdd')
p5, = host.plot(days[rng],gdd_ldm[rng],color=(1,0.5,0),label='ld_gdd')

p4, = par.plot(days,fdm.Veg,color=(0,1,0),label='fd_veg')
p6, = par.plot(days,ldm.Veg,color=(0,0.5,0),label='ld_veg')

p7, = par.plot(days,fdm.EGF,color=(0,0,1),label='fd_egf')
p9, = par.plot(days,ldm.EGF,color=(0,0,0.5),label='ld_egf')

p10, = par.plot(days,fdm.LGF,color=(0,1,1),label='fd_lgf')
p12, = par.plot(days,ldm.LGF,color=(0,0.5,0.5),label='ld_lgf')

host.set_xlim(75,325)
host.set_ylim(0,16)
par.set_ylim(0,250)
host.set_yticks(range(2,18,2))
host.set_yticklabels(['2','4','6','8(0)','10(2)','12(4)','14(6)','16(8)'])
par.set_yticks([0,50,100])
par.set_yticklabels(['0','50','100'])


host.set_xlabel('Day of Year')
host.set_ylabel('Degree Days [$^\circ$C day]')

par.text(347,100,'Development Phase [%]',rotation='vertical')

lines = [p1,p3,p4,p6,p7,p9,p10,p12,p2,p5]

t1, = plt.plot([],[],color=(1,0,0),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
t2, = plt.plot([],[],color=(0.5,0,0),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')

t3, = plt.plot([],[],color=(1,0.75,0.4),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
t4, = plt.plot([],[],color=(1,0.5,0),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')

v1, = plt.plot([],[],color=(0,1,0),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
v2, = plt.plot([],[],color=(0,0.5,0),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')

e1, = plt.plot([],[],color=(0,0,1),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
e2, = plt.plot([],[],color=(0,0,0.5),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')

l1, = plt.plot([],[],color=(0,1,1),marker='s',markersize=20,fillstyle='left',markeredgecolor='w',linestyle='none')
l2, = plt.plot([],[],color=(0,0.5,0.5),marker='s',markersize=20,fillstyle='right',markeredgecolor='w',linestyle='none')

plt.legend(((t1, t2), (t3, t4), (v1, v2), (e1,e2), (l1,l2)), 
	   ('KDD','GDD','Vegetative','Early Grain Fill','Late Grain Fill'),
            loc='upper left', bbox_to_anchor=(0.024,0.95), labelspacing=1.1, frameon=False)

plt.text(78,237,'Early|Recent',fontsize=9)

plt.savefig('region_gk_dev.pdf')
plt.close()

### County Plots
shpfile = '/Users/timaeus/Desktop/Projects/Data/UScounties/cb_2015_us_county_20m.shp'
cou = shpreader.Reader(shpfile)

shapename = 'admin_1_states_provinces_lakes_shp'
states_shp = shpreader.natural_earth(resolution='110m',
                                     category='cultural', name=shapename)
states = shpreader.Reader(states_shp)

# data to plot
cr2 = pd.read_csv('cr2.csv',index_col=0)
ctr = pd.read_csv('ctr.csv',index_col=0)
ctr_d = pd.read_csv('ctr_d.csv',index_col=0)
ctr_w = pd.read_csv('ctr_w.csv',index_col=0)
cbetas = pd.read_csv('cbetas.csv',index_col=0)

# yield trends, by GDD/KDD and weather/dev
gta = np.sum(ctr.iloc[:,0:3]*cbetas.iloc[:,0:3],axis=1)*10 
kta = np.sum(ctr.iloc[:,3:6]*cbetas.iloc[:,3:6],axis=1)*10
gtad = np.sum(ctr_d.iloc[:,0:3]*cbetas.iloc[:,0:3],axis=1)*10 
ktad = np.sum(ctr_d.iloc[:,3:6]*cbetas.iloc[:,3:6],axis=1)*10
gtaw = np.sum(ctr_w.iloc[:,0:3]*cbetas.iloc[:,0:3],axis=1)*10
ktaw = np.sum(ctr_w.iloc[:,3:6]*cbetas.iloc[:,3:6],axis=1)*10

tmp = np.asarray([gta,kta,gtad,ktad,gtaw,ktaw]).T

tr_av = pd.DataFrame(tmp,columns=['GDD','KDD','GDev','KDev','Gwea','Kwea'],
		     index=ctr.index)

#tr_av.to_csv('tr_av.csv')

cg = []
for i in cou.records():
  cg.append(i.attributes['GEOID'])

c_loc = list()

# need to make sure this is for the correct county set
geos = map(str,cr2.index)

for i,co in enumerate(cou.records()):
  if co.attributes['GEOID'] in geos:
    c_loc.append(co.attributes['GEOID'])

c_loc = map(int,c_loc)

cr2 = cr2.reindex(c_loc)
tr_av = tr_av.reindex(c_loc)
ctr = ctr.reindex(c_loc)
ctr_d = ctr_d.reindex(c_loc)
ctr_w = ctr_w.reindex(c_loc)
cbetas = cbetas.reindex(c_loc)

def cou_plot(y,rng,label,colbar,text_label,name,ori='horizontal'):
  cmap = plt.get_cmap(colbar)
  norm = mpl.colors.Normalize(rng[0],rng[1])
  col_prod = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
  
  ## plot boundaries
  ax = plt.axes(projection=ccrs.LambertConformal())
  ax.set_extent([-100.75, -81, 35.5, 49], ccrs.Geodetic())
  idx = 0 
  for i,county in enumerate(cou.geometries()):
    if cg[i] in geos:
      facecolor = col_prod.to_rgba(y.iloc[idx])[0:3]
      #edgecolor = 'black'
      idx += 1
      ax.add_geometries([county], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=facecolor,zorder=0)

  for s in states.geometries():
     ax.add_geometries([s], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black',zorder=1)

  if label != None: 
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array(y)
    cbar = plt.colorbar(sm,ax=ax,orientation=ori,pad=0.05,shrink=0.75,aspect=15)
    cbar.set_label(label)
  
  pte = plt.text(0.005,0.90,text_label,transform=ax.transAxes) 
 
  plt.savefig(name,bbox_inches='tight')
  plt.close()
  

cou_plot(cr2,[0,1],'R$^2$','gnuplot_r','','r2_all.pdf')
#cou_plot(cr2trs,[0,1],'R$^2$','','r2_st_tech.pdf')

#cou_plot(cr2,[0,1],'R$^2$','','r2.pdf')

# for plot labels
fn = ['Vegetative','Early Grain Fill','Late Grain Fill']

yrng = [-30,30]
trng = [-5,5]
krng = [-2.5,2.5]
for i in range(6):
  cyld = ctr.iloc[:,i]*cbetas.iloc[:,i]*1000
  cdyld = ctr_d.iloc[:,i]*cbetas.iloc[:,i]*1000
  cwyld = ctr_w.iloc[:,i]*cbetas.iloc[:,i]*1000
  cb = cbetas.iloc[:,i]*1000
  nm = str.split(cb.name,'_')
  nm[0] = fn[i%3]
  nm = ' '.join(nm)
  if i<3:
    cou_plot(cyld,yrng,nm+' Yield Trend [(t/ha)/decade]','RdYlGn','',cyld.name+'.pdf')
    cou_plot(cdyld,yrng,nm+' Yield Trend [(t/ha)/decade]','RdYlGn','',
							cdyld.name+'_D.pdf')
    cou_plot(cwyld,yrng,nm+' Yield Trend [(t/ha)/decade]','RdYlGn','',
							cwyld.name+'_W.pdf')
    cou_plot(ctr.iloc[:,i],trng,nm+' Trend [($^\circ$C day)/yr]','RdBu_r','',
       						cyld.name+'_T.pdf')
    cou_plot(ctr_w.iloc[:,i],trng,nm+' Trend [($^\circ$C day)/yr]','RdBu_r','',
							cyld.name+'_Tw.pdf')
    cou_plot(ctr_d.iloc[:,i],trng,nm+' Trend [($^\circ$C day)/yr]','RdBu_r','',
							cyld.name+'_Td.pdf')
 
  else:
    cou_plot(cyld,yrng,nm+' Yield Trend [(kg/ha)/yr]','RdYlGn','',cyld.name+'.pdf')
    cou_plot(cdyld,yrng,nm+' Yield Trend [(kg/ha)/yr]','RdYlGn','',
							cdyld.name+'_D.pdf')
    cou_plot(cwyld,yrng,nm+' Yield Trend [(kg/ha)/yr]','RdYlGn','',
							  cwyld.name+'_W.pdf')
    cou_plot(ctr.iloc[:,i],krng,nm+' Trend [($^\circ$C day)/yr]','RdBu_r','',
							      cyld.name+'_T.pdf')
    cou_plot(ctr_w.iloc[:,i],krng,nm+' Trend [($^\circ$C day)/yr]','RdBu_r','',
							cyld.name+'_Tw.pdf')
    cou_plot(ctr_d.iloc[:,i],krng,nm+' Trend [($^\circ$C day)/yr]','RdBu_r','',
							cyld.name+'_Td.pdf')

yd_mat = pd.read_csv('yd_mat.csv',index_col=[0,1])
# straighten out the column names
col = yd_mat.columns.values
col[0] = 'Year'
col[col=='geoid.1'] = 'geoid'
yd_mat.columns = col
# convert geoid to string to access both state and county components
yd_mat.geoid = map(str,yd_mat.geoid)

geo = map(int,np.unique(yd_mat.geoid))
cou_ytr = []
for g in geo:
  x = yd_mat.loc[g].Year
  y = yd_mat.loc[g].Yld
  cou_ytr.append(np.polyfit(x,y,1)[0]*10)

cou_ytr = pd.DataFrame(cou_ytr,index=ctr.index)
cou_ytr = cou_ytr.reindex(c_loc)

cou_plot(cou_ytr,[0,2],'Total Yield Trend [(t/ha)/decade]','YlGn','','tot_ytr.pdf','vertical')

gkrng = [-0.3,0.3]

cou_plot(tr_av.Gwea,gkrng,None,'RdYlGn','a)','gdd_wtry.pdf')
cou_plot(tr_av.Kwea,gkrng,None,'RdYlGn','b)','kdd_wtry.pdf')
cou_plot(tr_av.GDev,gkrng,None,'RdYlGn','c)','gdd_dtry.pdf')
cou_plot(tr_av.KDev,gkrng,None,'RdYlGn','d)','kdd_dtry.pdf')
cou_plot(tr_av.GDD,gkrng,None,'RdYlGn','e)','gdd_try.pdf')
cou_plot(tr_av.KDD,gkrng,None,'RdYlGn','f)','kdd_try.pdf')

# colorbar only
fig = plt.figure(figsize=(8, 2))
ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])
cmap = mpl.cm.RdYlGn
norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal')
cb.ax.tick_params(labelsize=18)
cb.set_label('Yield Trend [(t/ha)/decade]',size=18)
plt.savefig('yld_colbar.pdf')
plt.close('all')

st_all = ph_tr.index.get_level_values(0)

drng = [-1,1]

def st_plot(y,rng,label,colbar,text_label,name,ori='horizontal'):
  cmap = plt.get_cmap(colbar)
  norm = mpl.colors.Normalize(rng[0],rng[1])
  col_prod = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
  sg = list(states.geometries())  
  ## plot boundaries
  ax = plt.axes(projection=ccrs.LambertConformal())
  ax.set_extent([-100.75, -81, 35.5, 49], ccrs.Geodetic())
  for i,s in enumerate(states.records()):
    stn = s.attributes['name'].upper()
    if stn in st_all:
      facecolor = col_prod.to_rgba(y.loc[stn])[0:3]
      ax.add_geometries([sg[i]], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor='black',zorder=0)
    else:
      ax.add_geometries([sg[i]], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black',zorder=1)

  if label != None: 
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array(y)
    cbar = plt.colorbar(sm,ax=ax,orientation=ori,pad=0.05,shrink=0.75,aspect=15)
    cbar.set_label(label)
  
  pte = plt.text(0.005,0.90,text_label,transform=ax.transAxes) 
 
  plt.savefig(name,bbox_inches='tight')
  plt.close()

label = 'Vegetative Start Date Trend [days/year]'
st_plot(ph_tr.loc[:,'Veg_st'],drng,label,'RdYlGn','','Veg_st.pdf')
label = 'Early Grain Fill Start Date Trend [days/year]'
st_plot(ph_tr.loc[:,'EGF_st'],drng,label,'RdYlGn','','EGF_st.pdf')
label = 'Late Grain Fill Start Date Trend [days/year]'
st_plot(ph_tr.loc[:,'LGF_st'],drng,label,'RdYlGn','','LGF_st.pdf')
label = 'Vegetative End Date Trend [days/year]'
st_plot(ph_tr.loc[:,'Veg_en'],drng,label,'RdYlGn','','Veg_en.pdf')
label = 'Early Grain Fill End Date Trend [days/year]'
st_plot(ph_tr.loc[:,'EGF_en'],drng,label,'RdYlGn','','EGF_en.pdf')
label = 'Late Grain Fill End Date Trend [days/year]'
st_plot(ph_tr.loc[:,'LGF_en'],drng,label,'RdYlGn','','LGF_en.pdf')
label = 'Vegetative Duration Trend [days/year]'
st_plot(ph_tr.loc[:,'Veg'],drng,label,'RdYlGn','','Veg.pdf')
label = 'Early Grain Fill Duration Trend [days/year]'
st_plot(ph_tr.loc[:,'EGF'],drng,label,'RdYlGn','','EGF.pdf')
label = 'Late Grain Fill Duration Trend [days/year]'
st_plot(ph_tr.loc[:,'LGF'],drng,label,'RdYlGn','','LGF.pdf')

###############

# need to import tr_comb for this to run independently

## stacked yield plot
t = tr_comb[0]
yld = yld_comb.yld.values
tw = tr_comb[1]
td = tr_comb[2]
yrs = yld_comb.index
x = yrs - yrs[0]
y0 = np.polyfit(x,yld,1)[1]
yt = x*t
# base yield value to split between weather and adaptation
wy = yld_comb.Wea_GDD.values+yld_comb.Wea_KDD.values
dy = yld_comb.Dev_GDD.values+yld_comb.Dev_KDD.values
yld0 = np.array([y0]*len(x))
# offset remainder for weather and development
basey = np.polyfit(wy+dy,yld-yld0-yt,0)
# add base yield proportional to yield trend fraction
yldw = wy+basey*tw/(tw+td)
yldd = dy+basey*td/(tw+td)
#yld0 = np.array([np.mean(yld)+np.mean(yldw)+np.mean(yldd)]*36)
ylda = yld - yld0 - yt - yldw - yldd
y = np.stack([yld0,yt,yldd,yldw,ylda])
y = y.cumsum(axis=0)
# y-offset for clarity
# eps = 0.25
y1 = (x*t)+yld0
y2 = (x*td)+y1
y3 = (x*tw)+y2

plt.fill_between(yrs,0,y[0,:],facecolor='k',alpha=0.7,zorder=0)
plt.fill_between(yrs,y[0,:],y[1,:],facecolor='g',alpha=0.7,zorder=0)
plt.fill_between(yrs,y[1,:],y[2,:],facecolor='b',alpha=0.5,zorder=0)
plt.fill_between(yrs,y[2,:],y[3,:],facecolor='r',alpha=0.7,zorder=0)
#plt.fill_between(yrs,y[3,:],y[4,:],facecolor='k',alpha=0.5)
p0, = plt.plot(yrs,yld0,'k-',lw=2,label='Base Yield',zorder=1)
#p1, = plt.plot(yrs,y1,'g-',lw=2,label='Technology',zorder=2)
p1, = plt.plot(yrs,y1,'g-',lw=2,label='Other Factors',zorder=2)
p2, = plt.plot(yrs,y2,'b-',lw=2,label='Timing',zorder=2)
p3, = plt.plot(yrs,y3,'r-',lw=2,label='Climate',zorder=2)
p4, = plt.plot(yrs,y[3],color='0.66',lw=2,label='Yield Estimate',zorder=1)
p5, = plt.plot(yrs,yld,'k.',label='Yield Data',zorder=2)

#### make separate plot and concatenate?
# trend bars
yb = [yld0[-1],y1[-1],y2[-1],y3[-1]]
yb = np.stack([yb,yb])
plt.fill_between([2019,2020],yb[:,0],yb[:,1],facecolor='g',alpha=0.7)
plt.fill_between([2019,2020],yb[:,1],yb[:,2],facecolor='b',alpha=0.7)
plt.fill_between([2019,2020],yb[:,2],yb[:,3],facecolor='r',alpha=0.7)
# confidence intervals
yb95 = np.array(tr_comb95).ravel()*36
plt.plot([2019.25, 2019.25],yb[0,0]+yb95[0:2],'g')
plt.plot([2019.75, 2019.75],yb[0,1]+yb95[2:4],'b')
plt.plot([2019.25, 2019.25],yb[0,2]+yb95[4:6],'r')
plt.plot([2018,2018],[4,12],'k',lw=0.75)

plt.xlim([1981,2021])
plt.ylim([4,12])
plt.xlabel('Year')
plt.xticks(np.arange(1985,2020,5))
plt.ylabel('Yield [tonnes/ha]')
plt.legend(frameon=False,loc='upper left',numpoints=1,scatterpoints=1)
plt.savefig('yld_stack_of.pdf')

#imagegrid or subplot
yb = [yld0[-1],y1[-1],y2[-1],y3[-1]]
yb = np.stack([yb,yb])
plt.fill_between([0.05,.1],yb[:,0],yb[:,1],facecolor='g',alpha=0.7)
plt.fill_between([0.05,.1],yb[:,1],yb[:,2],facecolor='b',alpha=0.7)
plt.fill_between([0.05,.1],yb[:,2],yb[:,3],facecolor='r',alpha=0.7)
plt.xlim([0,1])
plt.ylim([4,12])
plt.axis('off')
plt.savefig('test.pdf')

# 3 alt
#plt.fill_between(yrs,0,y[0,:],facecolor='k',alpha=0.7)
#plt.fill_between(yrs,y[0,:],y[1,:],facecolor='g',alpha=0.7)
#plt.fill_between(yrs,y[1,:],y[2,:],facecolor='r',alpha=0.5)
#plt.fill_between(yrs,y[2,:],y[3,:],facecolor='b',alpha=0.7)
#p0, = plt.plot(0,yld0[0],'k-',lw=2,label='Base Yield')
#p1, = plt.plot(0,y1[0],'g-',lw=2,label='Technology')
#p2, = plt.plot(0,y2[0],'r-',lw=2,label='Weather')
#p3, = plt.plot(0,y3[0],'b-',lw=2,label='Adaptation')
#p4, = plt.plot(yrs,yld,'k.',label='Total Yield')
#
## trend bars
#yb = [yld0[-1],y1[-1],y2[-1],y3[-1]]
#yb = np.stack([yb,yb])
#plt.fill_between([2016.5,2017.5],yb[:,0],yb[:,1],facecolor='g',alpha=0.7)
#plt.fill_between([2016.5,2017.5],yb[:,1],yb[:,2],facecolor='r',alpha=0.7)
#plt.fill_between([2016.5,2017.5],yb[:,2],yb[:,3],facecolor='b',alpha=0.7)
## confidence intervals
#yb95 = np.array(tr_comb95).ravel()*35
#plt.plot([2016.75, 2016.75],yb[0,0]+yb95[0:2],'k')
#plt.plot([2017.25, 2017.25],yb[0,1]+yb95[2:4],'k')
#plt.plot([2016.75, 2016.75],yb[0,2]+yb95[4:6],'k')
#
#plt.xlim([1981,2018])
#plt.ylim([4,12])
#plt.xlabel('Year')
#plt.ylabel('Yield [tonnes/ha]')
#plt.legend(frameon=False,loc='upper left',numpoints=1,scatterpoints=1)
#plt.savefig('yld_stack_bar.pdf')

# Figure 4
# difference in yield from recent shifts in development
# first get the interquartile range for all the years
c_durdiff = pd.DataFrame(c_durdiff)
iqb = c_durdiff.boxplot(by='Year',return_type='dict')[0]['boxes']
iqr = [i.get_ydata()[[0,3]] for i in iqb]
iqr_mat = np.matrix(iqr)
plt.close()

#yrs = np.linspace(0,35,36)
x = np.linspace(1,37,37)
b = np.polyfit(ddcl.index.get_level_values('Year'),ddcl,1)
m_dd = c_durdiff.groupby('Year').mean().values
#b = np.polyfit(yrs,m_dd.values,1)
#yrx = np.array(c_durdiff.index.get_level_values('Year'))
#yrd = np.array(c_durdiff.values)
#b = np.polyfit(yrx,yrd,1)
#ix = np.in1d(m_dd.index,[1983,1988,2012],invert=True)
#bo = np.polyfit(yrs[ix],m_dd.values[ix],1)

plt.vlines(x,iqr_mat[:,0],iqr_mat[:,1],colors='grey')
plt.xlabel('Year')
plt.ylabel('Yield Difference [t/ha]')
plt.plot(x,m_dd,'ko')
plt.plot(x,b[1]+b[0]*(x+1980),'r')
#plt.plot(x,b[1]+b[0]*yrs,'r')
#plt.plot(x,bo[1]+bo[0]*yrs,'m')
plt.plot(x,trdd95[0],'r--')
plt.plot(x,trdd95[1],'r--')
#plt.plot(x,trddo95[0],'m--')
#plt.plot(x,trddo95[1],'m--')
locs = [1,5,10,15,20,25,30,35]
labels = ['1981','1985','1990','1995','2000','2005','2010','2015']
plt.xticks(locs,labels)

plt.savefig('yld_diff.pdf')
plt.close('all')

# by state
geoid = map(str,c_durdiff.index.get_level_values(0))
st = [g[0:2] for g in geoid]
for i,f in enumerate(fips):
  l = np.in1d(st,f)
  s = c_durdiff[l]
  sbox = s.boxplot(by='Year',showmeans=True,return_type='dict')[0]
  plt.close()

  sm = [s.get_ydata() for s in sbox['means']]
  sm = np.array(sm).ravel()
  iqr = [iq.get_ydata()[[0,3]] for iq in sbox['boxes']]
  iqr_mat = np.matrix(iqr)
  
  yrs = np.linspace(0,36,37)
  x = yrs+1
  
  plt.vlines(x,iqr_mat[:,0],iqr_mat[:,1],colors='grey')
  plt.xlabel('Year')
  plt.ylabel('Yield Difference [t/ha]')
  plt.plot(x,sm,'k.')
  b = np.polyfit(yrs,sm,1)
  plt.plot(x,b[1]+b[0]*yrs,'k-')
  locs = [1,5,10,15,20,25,30,35]
  labels = ['1981','1985','1990','1995','2000','2005','2010','2015']
  plt.xticks(locs,labels)
  plt.title(stn[i])
  #plt.ylim(-0.7,1.0)
   
  plt.savefig('yld_diff_'+stn[i]+'_var.pdf')
  plt.close('all')

# clustering
for i in range(775):                                          
  plt.plot(clus.lon.values[i],clus.lat.values[i],marker="$"+str(km108cls.labels_[i])+"$",color='k')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('clusters.pdf')
plt.close('all')
  

# boxplots
#fig,ax = plt.subplots()
#c_durdiff.boxplot(by='Year',whis=False,showfliers=False,ax=ax)
#ax.yaxis.grid(b=None)
#ax.xaxis.grid(b=None)
#plt.suptitle('')
#plt.xticks(rotation=0)
#plt.xlabel('Year')
#plt.ylabel('Yield Difference [t/ha]')
#plt.title('')
#ax.plot(x,m_dd.values,'ko')
#ax.plot(x,b[1]+b[0]*yrs,'r')
#ax.plot(x,trdd95[0],'r--')
#ax.plot(x,trdd95[1],'r--')
#ax.plot(x,trddo95[0],'m--')
#ax.plot(x,trddo95[1],'m--')
#locs = [1,5,10,15,20,25,30,35]
#labels = ['1981','1985','1990','1995','2000','2005','2010','2015']
#plt.xticks(locs,labels)
#
#plt.savefig('yld_diff_full2conf.pdf')
#plt.close('all')

#geo = np.unique(c_durdiff.index.get_level_values(0))
#c_ddiff_tr = np.zeros([841,2])
#for i,g in enumerate(geo):
#  c = c_durdiffA.loc[g]
#  x = c.index.values-1981
#  y = c.dd.values.ravel()
#  w = c.Area.values.ravel()
#  c_ddiff_tr[i,:] = np.polyfit(x,y,1,w=w)
#
#c_ddiff_tr = pd.DataFrame(c_ddiff_tr,index=geo,columns=['tr','int'])
#
## get in county plotting order
#c_ddiff_tr = c_ddiff_tr.reindex(c_loc)
#cou_plot(c_ddiff_tr.tr,[-0.005,0.005],'Duration Difference Yield Trend [t/ha]','RdYlGn','','c_ddiff_tr.pdf')
#
## quantile trend through yield differences?
#c_durdiff = pd.DataFrame(c_durdiff,columns = ['ydiff'])
#c_durdiff['year'] = c_durdiff.index.get_level_values(1)
#
#mod = smf.quantreg('ydiff ~ year',c_durdiff)
#
#quant = np.arange(0.05,0.96,0.05)
#
#for q in quant:
#  out = mod.fit(q=q)
#  qslope.append(out.params[1])
#
### no boxplot
#plt.plot(yrs,dur_diff,'ko',yrs,dur_tr[1]+dur_tr[0]*yrs,'r')
#
#plt.xlabel('Year')
#plt.ylabel('Yield Difference [tonnes/ha]')
#plt.savefig('yld_diff.pdf')

