# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:42:53 2021

@author: eckmb
"""

import mechanize
import json
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from time import sleep

# downloads dictionary of forecast id's and year/dates. zone from 0-9
def downloadZoneDictionary(zone):
    res = requests.get("https://www.avalanche.state.co.us/caic/pub_bc_avo.php?zone_id=%i" % (zone))
    match = re.search("(?<=_hs_options\['arc_bc_avo_fx_sel'\] = \[\n)(.*\n)", res.text)
    data_string = match.group().replace("\'", "\"")
    print(data_string)
    return json.loads(data_string)

def downloadAllZoneDictionaries(zones, delay):
    zone_dicts = []
    for i in range(len(zones)):
        zone_dicts.append(downloadZoneDictionary(i))
        sleep(delay)
    return zone_dicts

delay_time = 1

ratings = {'No Rating (-)': 0, 'Low (1)': 1, 'Moderate (2)':2, 'Considerable (3)': 3, 'High (4)':4, 'Extreme (5)': 5}
zones = ["Steamboat & Flat Tops", "Front Range", "Vail & Summit County", "Sawatch", "Aspen", "Gunnison", "Grand Mesa", "North San Juan", "South San Juan", "Sangre de Cristo"]

zone_dict = downloadAllZoneDictionaries(zones, delay_time)

f= open(r'C:\Users\eckmb\Downloads\data.json')

z1 = json.load(f)

# br.select_form(nr=0)

# br.form['arc_bc_avo_fx_sel[0]'] = [years[2]]
# br.form['arc_bc_avo_fx_sel[1]'] = [str(z1[years[2]]['values'][0])]
# br.form.fixup()
# req = br.submit()



# res = requests.get("https://www.avalanche.state.co.us/caic/pub_bc_avo.php?bc_avo_fx_id=%i" % (2140))

# soup = BeautifulSoup(res.content)
# print(soup.find_all('td', class_='fx-mtn-date'))


# for year in years:
#     for value in z1[year]['values']:

def getZoneDF(zone, zone_lookup, delay):
    data = pd.DataFrame(columns=["Zone", "Year", "Date", "Above Danger", "Near Danger", "Below Danger"])
    years = list(zone_lookup.keys())
    for year in years:
        for idx, value in enumerate(zone_lookup[year]['values']):
            print("Getting Value: %i from year: %s" % (value, year))
            res = requests.get("https://www.avalanche.state.co.us/caic/pub_bc_avo.php?bc_avo_fx_id=%i" % (value))
            soup = BeautifulSoup(res.content)
            above_danger = ratings[soup.find('td', class_=re.compile("above_danger.*$")).find('strong').get_text()]
            near_danger = ratings[soup.find('td', class_=re.compile("near_danger.*$")).find('strong').get_text()]
            below_danger = ratings[soup.find('td', class_=re.compile("below_danger.*$")).find('strong').get_text()]
            
            data = data.append({'Zone':zone,'Year':year, 'Date': zone_lookup[year]['texts'][idx], 'Above Danger':above_danger, 'Near Danger': near_danger,'Below Danger': below_danger}, ignore_index=True)
            sleep(delay)
        
    return data


z1data = getZoneDF(1, z1, delay_time)

zonedata = []
zonedata.append(getZoneDF(0, zone_dict[0], delay_time))
zonedata.append(z1data)
zonedata.append(getZoneDF(2, zone_dict[2], delay_time))
zonedata.append(getZoneDF(3, zone_dict[3], delay_time))
zonedata.append(getZoneDF(4, zone_dict[4], delay_time))
zonedata.append(getZoneDF(5, zone_dict[5], delay_time))
zonedata.append(getZoneDF(6, zone_dict[6], delay_time))
zonedata.append(getZoneDF(7, zone_dict[7], delay_time))
zonedata.append(getZoneDF(8, zone_dict[8], delay_time))
zonedata.append(getZoneDF(9, zone_dict[9], delay_time))

for idx, df in enumerate(zonedata):
    df.to_csv("Data/z%idata.csv" % (idx))