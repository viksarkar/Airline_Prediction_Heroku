import streamlit as st
import pandas as pd
import numpy as np
import pickle
#import base64
#import seaborn as sns
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime


st.write("""
## Airline Fullness (Load Factor) Prediction App
""")
st.write("""
Don\'t get cramped, let us help predict how full your flight will be. Just choose your origin, destination, \
month of flight, and (optionally) the carrier and we will let you know how full to expect your flight to be!
""")

def plotcarriergraph(allcarrierdata):
    xvals=[]
    yvals=[]
    carriers = allcarrierdata['Carrier'].unique()
    for carriernum in range(len(carriers)):
        currentcarrier = carriers[carriernum]
        datasubset = allcarrierdata[allcarrierdata['Carrier']==currentcarrier]
        for carriervalue in range(datasubset.shape[0]):
            xvals.append(carriernum+0.5)
            yvals.append(datasubset.iloc[carriervalue]['Load Factor (%)'])
    fig = plt.figure(figsize=(6,6))
    plt.plot(xvals, yvals, 'xr')
    plt.ylabel("Fullness (%)", fontsize=14)
    plt.title("Fullness of all possible flights found on this route \n", fontsize=20)
    plt.xlim(0, len(carriers))
    plt.xticks(np.arange(0.5,len(carriers)+0.5), carriers, rotation=30, fontsize=12)
    fig.savefig('CarrierFig.jpg', dpi=300, bbox_inches='tight')    
    return 0

Origins = pd.read_csv("Top_Airports.csv", header=None)
origin_choice = st.sidebar.selectbox('Select your Origin Airport:', Origins, index=76)

Destinations = pd.read_csv("Top_Airports.csv", header=None)
dest_choice = st.sidebar.selectbox('Select your Destination Airport:', Destinations, index=73)

months = ['January','February','March','April','May','June','July','August',\
          'September','October','November','December']
travel_month = st.sidebar.selectbox('Select your month of travel', months)
datetime_object = datetime.datetime.strptime(travel_month, "%B")
month_choice = datetime_object.month

Carriers = pd.read_csv("Top_Carriers.csv", header=None)
carrier_choice = st.sidebar.selectbox('Select your Carrier (OPTIONAL):', Carriers, index=0)

data = pd.read_csv('All_Flights.csv')
allcarrierdata = data[(data['ORIGIN']==origin_choice) & (data['DEST']==dest_choice) &\
                   (data['MONTH']==month_choice)]
numflights = allcarrierdata.shape[0]
carriers = allcarrierdata['UNIQUE_CARRIER_NAME'].unique()
numcarriers = len(carriers)
singlecarrierdata = allcarrierdata[allcarrierdata['UNIQUE_CARRIER_NAME']==carrier_choice]
numflightssinglecarrier = singlecarrierdata.shape[0]

if numflights == 0:
    st.write('We have unfortunately found no flights in our database between',\
             origin_choice,'and', dest_choice, 'for travel in',\
             travel_month,'.')
elif carrier_choice == 'No Selection':
    st.write('We have found', numflights, 'flights between', origin_choice,'and'\
             , dest_choice, 'on', numcarriers,'unique carriers for travel in',\
             travel_month,'.')
elif numflightssinglecarrier == 0:
    st.write('We have found', numflights, 'flights between', origin_choice, 'and'\
             , dest_choice, 'on', numcarriers, 'unique carriers for travel in',\
             travel_month,', but none are on', carrier_choice,'.')    
else:
    st.write('We have found', numflights, 'flights between', origin_choice, 'and'\
             , dest_choice, 'on', numcarriers, 'unique carriers for travel in',\
             travel_month,', including', numflightssinglecarrier, 'on',\
             carrier_choice,'.')

encoder = pickle.load(open('OneHotEncoder.pkl', 'rb'))
rfr = pickle.load(open('Load_Factor_SVR_Model.pkl', 'rb'))

if numflights == 0:
    image_hyperlink = 'http://www.gcmap.com/map?P='+origin_choice+'-'+dest_choice+'&MS=bm&MR=540&MX=540x540&PM=b:disc7%2b%22%25t%25+%28N%22'
    st.image(image_hyperlink, use_column_width=True)
    st.write('Map source : www.gcmap.com')
elif ((numflights != 0) and (carrier_choice!='No Selection') and (numflightssinglecarrier==0)):
    X = encoder.transform(allcarrierdata)
    planetypes = pd.read_csv('L_AIRCRAFT_TYPE.csv')
    allcarrierdata = allcarrierdata.merge(planetypes,how='left',left_on='AIRCRAFT_TYPE',right_on='Code')
    allcarrierdata['Load Factor (%)'] = rfr.predict(X)
    allcarrierdata['Load Factor (%)']=round(allcarrierdata['Load Factor (%)'],5)
    allcarrierdata = allcarrierdata.rename(columns={"UNIQUE_CARRIER_NAME":"Carrier",\
                                                    "Description": "Aircraft Type",\
                                                    "DEPARTURES_SCHEDULED": "No. of Flights"})
    plotcarriergraph(allcarrierdata)
    col1, col2 = st.columns(2)
    image_hyperlink = 'http://www.gcmap.com/map?P='+origin_choice+'-'+dest_choice+'&MS=bm&MR=540&MX=540x540&PM=b:disc7%2b%22%25t%25+%28N%22'
    col1.image(image_hyperlink, use_column_width=True)
    col1.write('Map source : www.gcmap.com')
    col2.image('CarrierFig.jpg', use_column_width=True)
elif ((numflights != 0) and (carrier_choice!='No Selection') and (numflightssinglecarrier!=0)):
    X = encoder.transform(allcarrierdata)
    planetypes = pd.read_csv('L_AIRCRAFT_TYPE.csv')
    allcarrierdata = allcarrierdata.merge(planetypes,how='left',left_on='AIRCRAFT_TYPE',right_on='Code')
    allcarrierdata['Load Factor (%)'] = rfr.predict(X)
    allcarrierdata['Load Factor (%)']=round(allcarrierdata['Load Factor (%)'],5)
    allcarrierdata = allcarrierdata.rename(columns={"UNIQUE_CARRIER_NAME":"Carrier",\
                                                    "Description": "Aircraft Type",\
                                                    "DEPARTURES_SCHEDULED": "No. of Flights"})
    plotcarriergraph(allcarrierdata)
    finaldata = allcarrierdata[allcarrierdata['Carrier']==carrier_choice]
    finaldata = finaldata[['Carrier','Aircraft Type','No. of Flights','Load Factor (%)']]
    finaldata['No. of Flights'] = finaldata['No. of Flights'].astype(int)
    finaldata = finaldata.sort_values('Load Factor (%)', ascending=True)
    finaldata['Load Factor (%)'] = finaldata['Load Factor (%)'].astype(int)
    st.table(finaldata.assign(hack='').set_index('hack'))
    col1, col2 = st.columns(2)
    image_hyperlink = 'http://www.gcmap.com/map?P='+origin_choice+'-'+dest_choice+'&MS=bm&MR=540&MX=540x540&PM=b:disc7%2b%22%25t%25+%28N%22'
    col1.image(image_hyperlink, use_column_width=True)
    col1.write('Map source : www.gcmap.com')
    col2.image('CarrierFig.jpg', use_column_width=True)
else:
    X = encoder.transform(allcarrierdata)
    planetypes = pd.read_csv('L_AIRCRAFT_TYPE.csv')
    allcarrierdata = allcarrierdata.merge(planetypes,how='left',left_on='AIRCRAFT_TYPE',right_on='Code')
    allcarrierdata['Load Factor (%)'] = rfr.predict(X)
    allcarrierdata['Load Factor (%)']=round(allcarrierdata['Load Factor (%)'],5)
    allcarrierdata = allcarrierdata.rename(columns={"UNIQUE_CARRIER_NAME":"Carrier",\
                                                    "Description": "Aircraft Type",\
                                                    "DEPARTURES_SCHEDULED": "No. of Flights"})
    plotcarriergraph(allcarrierdata)
    finaldata = allcarrierdata[['Carrier','Aircraft Type','No. of Flights','Load Factor (%)']]
    finaldata['No. of Flights'] = finaldata['No. of Flights'].astype(int)
    finaldata = finaldata.sort_values('Load Factor (%)', ascending=True)
    finaldata['Load Factor (%)'] = finaldata['Load Factor (%)'].astype(int)
    st.table(finaldata.assign(hack='').set_index('hack'))
    col1, col2 = st.columns(2)
    image_hyperlink = 'http://www.gcmap.com/map?P='+origin_choice+'-'+dest_choice+'&MS=bm&MR=540&MX=540x540&PM=b:disc7%2b%22%25t%25+%28N%22'
    col1.image(image_hyperlink, use_column_width=True)
    col1.write('Map source : www.gcmap.com')
    col2.image('CarrierFig.jpg', use_column_width=True)
