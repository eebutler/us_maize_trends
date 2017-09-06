import requests, json
import pandas as pd
import numpy as np

# don't forget to delete the api_key before uploading...
api_key = '49BFB38E-61E8-3809-8C09-11776677625B'

# states to access
states = ['IL','IN','IA','KS','KY','MI','MN','MO','NE','OH','SD','WI']

# year range, remember python indexing
years = map(str,range(1981,2017))

# prepare an empty dataframe to accumulate all the values
df_all = pd.DataFrame()

# options
# 1) api/api_GET : retrieve values
# 2) api/get_param_values : possible fields for a given parameter
# payload = {'key':api_key,'param':{parameter}} i.e. 'source_desc'
# 3) api/get_counts : number of values, max for download is 50000
#url = 'http://quickstats.nass.usda.gov/api/get_param_values/'
url = 'http://quickstats.nass.usda.gov/api/api_GET/'
# loop through states to keep query size managable

payload = {'key':api_key,
	   'source_desc':'SURVEY',
	   'commodity_desc':'CORN',
	   'util_practice_desc':'GRAIN',
	   'statisticcat_desc':'YIELD',
	   'agg_level_desc':'COUNTY',
	   'year':years,
	   'format':'JSON',
	   'unit_desc':'BU / ACRE',
	   'prodn_practice_desc':'ALL PRODUCTION PRACTICES'}

for s in states:
    payload['state_alpha'] = s
    r = requests.get(url,params=payload)
    
    # get the json data into a pandas dataframe
    df = pd.read_json(json.dumps(r.json()),orient='split')
    
    # strip down to the fields we want: 
    # name, year, state_id, county_id, yield
    df_trunc = df[['county_name','year','state_fips_code',
        	 'county_code','Value']]
    
    
    df_all = pd.concat([df_all,df_trunc])
    print s

# get df_all in the correct order, after concatenation messes it up...
df_all = df_all[['county_name','year','state_fips_code',
  		 'county_code','Value']]

df_all.to_csv('yld.csv',header=False,index=False)

## area planted
del df_all
df_all = pd.DataFrame()
del payload['util_practice_desc']
del payload['unit_desc']
payload['statisticcat_desc'] = 'AREA PLANTED'

for s in states:
  payload['state_alpha'] = s
  r = requests.get(url,params=payload)
    
  # get the json data into a pandas dataframe
  df = pd.read_json(json.dumps(r.json()),orient='split')
  
  # strip down to the fields we want: 
  # name, year, state_id, county_id, yield
  df_trunc = df[['county_name','year','state_fips_code',
      	 'county_code','Value']]
  
  
  df_all = pd.concat([df_all,df_trunc])
  print s

df_all = df_all[['county_name','year','state_fips_code',
  		 'county_code','Value']]

# convert comma separated values into integers
area = list(df_all['Value'])
area = [a.replace(',','') for a in area]
df_all['Value'] = area

df_all.to_csv('pl_area.csv',header=False,index=False)

###### Development Data #######
payload = {'key':api_key,
	   'source_desc':'SURVEY',
	   'commodity_desc':'CORN',
	   'statisticcat_desc':'PROGRESS',
	   'agg_level_desc':'STATE',
	   'util_practice_desc':'ALL UTILIZATION PRACTICES',
	   'year':years,
	   'state_alpha':states,
	   'format':'JSON'}

r = requests.get(url,params=payload)

# get the json data into a pandas dataframe
df = pd.read_json(json.dumps(r.json()),orient='split')

# retrieve harvest for grain only 
payload['util_practice_desc'] = 'GRAIN'
r = requests.get(url,params=payload)
dfH = pd.read_json(json.dumps(r.json()),orient='split')

# concatentate
df = pd.concat([df,dfH])

# get progress data for each phase, run on individual states
def prg_rd(df):
  # empty Dataframe to initialize
  prg = pd.DataFrame()

  phas = ['PCT PLANTED', 'PCT SILKING', 'PCT DOUGH',
	  'PCT DENTED', 'PCT MATURE', 'PCT HARVESTED']

  for p in phas:
    # pick out each phase
    pl = df['unit_desc'] == p 
    dfp = df[['week_ending','Value']][pl] 
    # build temporal index
    dt = []
    for i in dfp.index:
      dt.append(pd.datetime.strptime(dfp.ix[i,'week_ending'],'%Y-%m-%d'))
    dfp.index = pd.Index(dt)
    # get unique years to interpolate values
    yr = np.unique(dfp.index.year)
    allyr = pd.Series()
    for y in yr:
      val = list(dfp.loc[str(y)].Value)
      ind = dfp.loc[str(y)].index
      # ensure that each year begins at 0 and ends at 100
      val.insert(0,0)
      val.append(100)
      # add weeks to index for new values
      owk = pd.DateOffset(weeks=1)
      ind = ind.insert(0,ind[0]-owk)
      ind = ind.insert(len(ind),ind[-1]+owk)
      # convert to Series
      tmp = pd.Series(val, index=ind)
      # extend to full year, daily timestep
      dat_rng = pd.date_range(start='01-01-'+str(y),end='12-31-'+str(y))
      tmp = pd.Series(tmp, index = dat_rng)
      # add in initial zero value and interpolate
      tmp[0] = 0
      tmp = tmp.interpolate()
      allyr = pd.concat([allyr,tmp])

    # progress dataframe
    prg = pd.concat([prg,allyr],axis=1)
  # add column names
  #prg.columns = phas

  return prg

# states in data
st = np.unique(df.state_name)

allst = pd.DataFrame()
lst = []
for s in st:
  sdf = df[df.state_name == s]
  spr = prg_rd(sdf)
  # save number of instances for each state
  lst.append(len(spr))
  
  allst = pd.concat([allst,spr])

# build multi-index 
stnm = [st[s] for s in range(len(st)) for i in range(lst[s])] 
allst = pd.DataFrame(allst.values,index=[stnm,allst.index])
# re-assign column names
allst.columns = ['Planted','Silking','Dough','Dent','Mature','Harvest']

# may need to explicitly assign NaNs in output
allst.to_csv('progress.csv')


####### Irrigated and Harvested Area #######

payload = {'key':api_key,
	   'source_desc':'CENSUS',
	   'commodity_desc':'CORN',
	   'util_practice_desc':'GRAIN',
	   'statisticcat_desc':'AREA HARVESTED',
	   'agg_level_desc':'COUNTY',
	   'unit_desc':'ACRES',
	   'prodn_practice_desc':'ALL PRODUCTION PRACTICES',
	   'domain_desc':'TOTAL',
	   'state_alpha':states,
	   'format':'JSON'}
	  
r = requests.get(url,params=payload)

# get the json data into a pandas dataframe
df = pd.read_json(json.dumps(r.json()),orient='split')

area = list(df['Value'])
for i,a in enumerate(area):
    area[i] = a.replace(',','')

area = pd.Series(area,index=df.index)
# remove single farm counties
pl = area == '                 (D)'
area = area[~pl]

df['Value'] = area

df_trunc = df[['county_name','year',
	       'state_fips_code','county_code','Value']]

df_trunc.to_csv('harv_area.csv',header=False,index=False)

# set payload for irrigated area
payload['prodn_practice_desc'] = 'IRRIGATED'

r = requests.get(url,params=payload)

# get the json data into a pandas dataframe
df = pd.read_json(json.dumps(r.json()),orient='split')

area = list(df['Value'])
for i,a in enumerate(area):
    area[i] = a.replace(',','')

area = pd.Series(area,index=df.index)
# remove single farm counties
pl = area == '                 (D)'
area = area[~pl]

df['Value'] = area

df_trunc = df[['county_name','year',
	       'state_fips_code','county_code','Value']]

df_trunc.to_csv('irr_area.csv',header=False,index=False)
