#!/usr/bin/env python

import base64
import datetime
import json

# import os
import requests
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import sys
import os

# %%

fildir = "/sand/usgs/users/dnowacki/waves/"

if len(sys.argv) == 1:
    site = "bel"
else:
    site = sys.argv[1]

if len(sys.argv) == 2:
    apikey = "sofar.apikey"
else:
    apikey = sys.argv[2]

print(site)

deviceid = {
    "bel": "SPOT-0713",
    "guam-orig": "SPOT-1291",
    "bel-orig": "SPOT-0206",
    "eden": "SPOT-1438",
    "rich": "SPOT-1440",
    "peta": "SPOT-1432",
    "hunt": "SPOT-1437",
    "treas": "SPOT-1439",
    "guam": "SPOT-1743",
}
timestart = {
    "bel": "2020-11-25T00:00:00",
    "guam-orig": "2022-01-21T00:00:00",
    "bel-orig": "2020-02-01T00:00:00",
    "eden": "2021-10-20T00:00:00",
    "rich": "2021-10-20T00:00:00",
    "peta": "2021-10-20T00:00:00",
    "hunt": "2021-10-20T00:00:00",
    "treas": "2021-10-20T00:00:00",
    "guam": "2022-06-23T00:00:00",
}
title = {
    "bel": "Bellingham Bay Spotter Buoy",
    "guam-orig": "Guam Smart Mooring (original unit)",
    "bel-orig": "Bellingham Bay Spotter Buoy (original unit)",
    "eden": "San Francisco Bay: EDEN Spotter Buoy",
    "rich": "San Francisco Bay: RICH Spotter Buoy",
    "peta": "San Francisco Bay: PETA Spotter Buoy",
    "hunt": "San Francisco Bay: HUNT Spotter Buoy",
    "treas": "San Francisco Bay: TREAS Spotter Buoy",
    "guam": "Guam Spotter Buoy",
}
is_smartmooring = {
    "bel": False,
    "guam-orig": True,
    "bel-orig": False,
    "eden": False,
    "rich": False,
    "peta": False,
    "hunt": False,
    "treas": False,
    "guam": False,
}

headers = {}
with open(apikey) as f:
    headers["token"] = f.read().strip()

params = {
    "spotterId": deviceid[site],
    "limit": "500",
    "startDate": timestart[site],
    "includeWaves": "true",
    "includeWindData": "false",
    "includeSurfaceTempData": "false",
    "includeFrequencyData": "false",
    "includeDirectionalMoments": "false",
    "includePartitionData": "false",
}

try:
    dsold = xr.load_dataset(fildir + site + ".nc")
    params["startDate"] = str(dsold.time[-1].values)
    print(
        "starting from incremental nc file. First burst",
        dsold["time"][0].values,
        "last burst",
        dsold["time"][-1].values,
    )
except FileNotFoundError:
    dsold = xr.Dataset()
    params["startDate"] = timestart[site]
    print("starting from scratch with nc file")

print("params['startDate']", params["startDate"])

api_url = "https://api.sofarocean.com/api/wave-data"
response = requests.get(url=api_url, headers=headers, params=params)

lines = np.array(response.json()["data"]["waves"])

# get unique times
bigx = [x["timestamp"] for x in lines]
_, index = np.unique(bigx, return_index=True)

dsnew = xr.Dataset()
dsnew["time"] = xr.DataArray(
    pd.DatetimeIndex([x["timestamp"] for x in lines[index]]), dims="time"
)
dsnew["time"] = pd.DatetimeIndex(dsnew["time"].values)
for k in lines[0].keys():
    if k != "timestamp":
        dsnew[k] = xr.DataArray([x[k] for x in lines[index]], dims="time")


def smartmooring():
    smart_url = "https://api.sofarocean.com/api/sensor-data"
    smart_response = requests.get(url=smart_url, headers=headers, params=params)
    lines = smart_response.json()["data"]

    mptime = []
    mpval = []
    mttime = []
    mtval = []
    mttime1 = []
    mtval1 = []
    for line in lines:
        if (
            line["data_type_name"] == "rbrcoda3_meanpressure_21bits"
            and line["sensorPosition"] == 2
        ):
            mptime.append(line["timestamp"])
            mpval.append(line["value"])
        if (
            line["data_type_name"] == "rbrcoda3_meantemperature_20bits"
            and line["sensorPosition"] == 2
        ):
            mttime.append(line["timestamp"])
            mtval.append(line["value"])
        if (
            line["data_type_name"] == "rbrcoda3_meantemperature_20bits"
            and line["sensorPosition"] == 1
        ):
            mttime1.append(line["timestamp"])
            mtval1.append(line["value"])
    meanpres = xr.Dataset()
    meanpres["time"] = xr.DataArray([np.datetime64(x) for x in mptime], dims="time")
    meanpres["rbrcoda3_meanpressure_21bits"] = xr.DataArray(
        np.array(mpval) * 0.00001, dims="time"
    )
    meanpres["rbrcoda3_meanpressure_21bits"].attrs["units"] = "dbar"

    meantemp = xr.Dataset()
    meantemp["time"] = xr.DataArray([np.datetime64(x) for x in mttime], dims="time")
    meantemp["rbrcoda3_meantemperature_20bits_lower"] = xr.DataArray(
        np.array(mtval), dims="time"
    )
    meantemp["rbrcoda3_meantemperature_20bits_lower"].attrs["units"] = "degree_C"

    meantemp1 = xr.Dataset()
    meantemp1["time"] = xr.DataArray([np.datetime64(x) for x in mttime1], dims="time")
    meantemp1["rbrcoda3_meantemperature_20bits_upper"] = xr.DataArray(
        np.array(mtval1), dims="time"
    )
    meantemp1["rbrcoda3_meantemperature_20bits_upper"].attrs["units"] = "degree_C"

    return xr.merge([meantemp, meanpres, meantemp1])


dsnew["time"] = dsnew.time.dt.round("1min")

if is_smartmooring[site]:
    sm = smartmooring()
    sm["time"] = pd.to_datetime(sm["time"])
    sm["time"] = sm["time"].dt.round("1min")
    dsnew = xr.merge([dsnew, sm.reindex_like(dsnew)])

ds = xr.merge([dsold, dsnew])

# hacky fix for some bad times that make erddap fail. Ends up droppping two samples with xx:xx:30 timestamps
ds = ds.where(ds.time.dt.second != 30, drop=True)

for k in ds.data_vars:
    if "time" in ds[k].dims:
        ds[k][ds[k] == -9999] = np.nan

ds.attrs["title"] = title[site] + ". PROVISIONAL DATA SUBJECT TO REVISION."
ds.attrs["history"] = "Generated using sofar-buoy.py"

# ds['latitude'] = xr.DataArray([latlon[site]['lat']], dims='latitude')
# ds['longitude'] = xr.DataArray([latlon[site]['lon']], dims='longitude')

ds["feature_type_instance"] = xr.DataArray(site)
ds["feature_type_instance"].attrs["long_name"] = "station code"
ds["feature_type_instance"].attrs["cf_role"] = "timeseries_id"

ds.attrs["naming_authority"] = "gov.usgs.cmgp"
ds.attrs["original_folder"] = "waves"
ds.attrs["featureType"] = "timeSeries"
ds.attrs["cdm_timeseries_variables"] = "feature_type_instance, latitude, longitude"


def add_standard_attrs(ds):
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["institution"] = "U.S. Geological Survey"

    ds["time"].attrs["standard_name"] = "time"

    ds["significantWaveHeight"].attrs[
        "standard_name"
    ] = "sea_surface_wave_significant_height"
    ds["significantWaveHeight"].attrs["units"] = "m"

    ds["peakPeriod"].attrs[
        "standard_name"
    ] = "sea_surface_wave_period_at_variance_spectral_density_maximum"
    ds["peakPeriod"].attrs["units"] = "s"

    ds["meanPeriod"].attrs["standard_name"] = "sea_surface_wave_mean_period"
    ds["meanPeriod"].attrs["units"] = "s"

    ds["peakDirection"].attrs[
        "standard_name"
    ] = "sea_surface_wave_from_direction_at_variance_spectral_density_maximum"
    ds["peakDirection"].attrs["units"] = "degree"

    ds["peakDirectionalSpread"].attrs[
        "standard_name"
    ] = "sea_surface_wave_directional_spread_at_variance_spectral_density_maximum"
    ds["peakDirectionalSpread"].attrs["units"] = "degree"

    ds["meanDirection"].attrs["standard_name"] = "sea_surface_wave_from_direction"
    ds["meanDirection"].attrs["units"] = "degree"

    ds["meanDirectionalSpread"].attrs[
        "standard_name"
    ] = "sea_surface_wave_directional_spread"
    ds["meanDirectionalSpread"].attrs["units"] = "degree"

    if "latitude" in ds:
        ds["latitude"].attrs["long_name"] = "latitude"
        ds["latitude"].attrs["units"] = "degrees_north"
        ds["latitude"].attrs["standard_name"] = "latitude"
        ds["latitude"].encoding["_FillValue"] = None

    if "longitude" in ds:
        ds["longitude"].attrs["long_name"] = "longitude"
        ds["longitude"].attrs["units"] = "degrees_east"
        ds["longitude"].attrs["standard_name"] = "longitude"
        ds["longitude"].encoding["_FillValue"] = None


add_standard_attrs(ds)

ds = ds.squeeze()

# %%
# make a backup
now = datetime.datetime.now()
timestr = now.strftime("%Y%m%d%H%M%S")
hour = now.strftime("%H")
try:
    if hour == "00":
        shutil.copy(
            fildir + site + ".nc", fildir + "../waves_bak/" + site + timestr + ".nc"
        )
except:
    print("Could not make backup. This may occur on first run")
ds.to_netcdf(fildir + site + ".nc", encoding={"time": {"dtype": "int32"}})
