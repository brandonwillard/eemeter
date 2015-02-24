import ftplib
import StringIO
import gzip
import os
import json
from datetime import datetime
from datetime import timedelta
import numpy as np
import requests
from . import ureg, Q_
from pkg_resources import resource_stream

class WeatherSourceBase:
    def get_average_temperature(self,consumptions,unit_name):
        """Returns a list of average temperatures of each consumption period in
        the given unit (usually "degF" or "degC").
        """
        unit = ureg.parse_expression(unit_name)
        avg_temps = []
        for consumption in consumptions:
            avg_temps.append(self.get_consumption_average_temperature(consumption,unit))
        return avg_temps

    def get_consumption_average_temperature(self,consumption,unit):
        """Returns the average temperature during the duration of a single
        Consumption instance as calculated by taking the mean of daily average
        temperatures.
        """
        avg_temps = self.get_consumption_daily_temperatures(consumption,unit)
        return np.mean(avg_temps)

    def get_daily_temperatures(self,consumptions,unit_name):
        """Returns, for each consumption, a list of average daily temperatures
        observed during the duration of that consumption period. Return value
        is a list of lists of temperatures.
        """
        unit = ureg.parse_expression(unit_name)
        daily_temps = []
        for consumption in consumptions:
            daily_temps.append(self.get_consumption_daily_temperatures(consumption,unit))
        return daily_temps

    def get_consumption_daily_temperatures(self,consumption,unit):
        """Returns, for a particular consumption instance, a list of average
        daily temperatures observed during the duration of that consumption
        period. Result is a list of temperatures.
        """
        avg_temps = []
        for days in xrange(consumption.timedelta.days):
            day = consumption.start + timedelta(days=days)
            temp = self.get_daily_average_temperature(day,unit)
            avg_temps.append(temp)
        return avg_temps

    def get_daily_average_temperature(self,day,unit):
        """Should return the average temperature of the given day. Must be
        implemented by inheriting classes.
        """
        raise NotImplementedError

    def get_hdd(self,consumptions,unit_name,base):
        """Returns, for each consumption, the total heating degree days
        observed during the consumption period.
        """
        unit = ureg.parse_expression(unit_name)
        hdds = []
        for consumption in consumptions:
            hdds.append(self.get_consumption_hdd(consumption,unit,base))
        return hdds

    def get_consumption_hdd(self,consumption,unit,base):
        """Returns the total heating degree days observed during the
        consumption period.
        """
        total_hdd = 0.
        for days in xrange(consumption.timedelta.days):
            day = consumption.start + timedelta(days=days)
            temp = self.get_daily_average_temperature(day,unit)
            if temp < base:
                total_hdd += base - temp
        return total_hdd

    def get_cdd(self,consumptions,unit_name,base):
        """Returns, for each consumption, the total cooling degree days
        observed during the consumption period.
        """
        unit = ureg.parse_expression(unit_name)
        cdds = []
        for consumption in consumptions:
            cdds.append(self.get_consumption_cdd(consumption,unit,base))
        return cdds

    def get_consumption_cdd(self,consumption,unit,base):
        """Returns the total cooling degree days observed during the
        consumption period.
        """
        total_cdd = 0.
        for days in xrange(consumption.timedelta.days):
            day = consumption.start + timedelta(days=days)
            temp = self.get_daily_average_temperature(day,unit)
            if temp > base:
                total_cdd += temp - base
        return total_cdd

class GSODWeatherSource(WeatherSourceBase):
    def __init__(self,station_id,start_year,end_year):
        if len(station_id) == 6:
            # given station id is the six digit code, so need to get full name
            with resource_stream('eemeter.resources','GSOD-ISD_station_index.json') as f:
                station_index = json.load(f)
            # take first station in list
            potential_station_ids = station_index[station_id]
        else:
            # otherwise, just use the given id
            potential_station_ids = [station_id]
        self._data = {}
        self._source_unit = ureg.degF
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()
        data = []
        for year in xrange(start_year,end_year + 1):
            string = StringIO.StringIO()
            # not every station will be available in every year, so use the
            # first one that works
            for station_id in potential_station_ids:
                try:
                    ftp.retrbinary('RETR /pub/data/gsod/{year}/{station_id}-{year}.op.gz'.format(station_id=station_id,year=year),string.write)
                    break
                except (IOError,ftplib.error_perm):
                    pass
            string.seek(0)
            f = gzip.GzipFile(fileobj=string)
            self._add_file(f)
            string.close()
            f.close()
        ftp.quit()

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance.
        """
        return self._data[day.strftime("%Y%m%d")].to(unit).magnitude

    def _add_file(self,f):
        for line in f.readlines()[1:]:
            columns=line.split()
            self._data[columns[2]] = Q_(float(columns[3]),self._source_unit)

class ISDWeatherSource(WeatherSourceBase):
    def __init__(self,station_id,start_year,end_year):
        if len(station_id) == 6:
            # given station id is the six digit code, so need to get full name
            with resource_stream('eemeter.resources','GSOD-ISD_station_index.json') as f:
                station_index = json.load(f)
            # take first station in list
            potential_station_ids = station_index[station_id]
        else:
            # otherwise, just use the given id
            potential_station_ids = [station_id]
        self._data = {}
        self._source_unit = ureg.degC
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()
        data = []
        for year in xrange(start_year,end_year + 1):
            string = StringIO.StringIO()
            # not every station will be available in every year, so use the
            # first one that works
            for station_id in potential_station_ids:
                try:
                    ftp.retrbinary('RETR /pub/data/noaa/{year}/{station_id}-{year}.gz'.format(station_id=station_id,year=year),string.write)
                    break
                except (IOError,ftplib.error_perm):
                    pass
            string.seek(0)
            f = gzip.GzipFile(fileobj=string)
            self._add_file(f)
            string.close()
            f.close()
        ftp.quit()

    def get_consumption_average_temperature(self,consumption,unit):
        """Gets the average temperature during a particular Consumption
        instance. Resolution limit: hourly.
        """
        avg_temps = []
        null = Q_(float("nan"),self._source_unit)
        n_hours = consumption.timedelta.days * 24 + consumption.timedelta.seconds // 3600
        for hours in xrange(n_hours):
            hour = consumption.start + timedelta(seconds=hours*3600)
            hourly = self._data.get(hour.strftime("%Y%m%d%H"),null).to(unit).magnitude
            avg_temps.append(hourly)
        # mask nans
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        null = Q_(float("nan"),self._source_unit)
        day_str = day.strftime("%Y%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self._data.get("{}{:02d}".format(day_str,i),null).to(unit).magnitude
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def _add_file(self,f):
        for line in f.readlines():
            # line[4:10] # USAF
            # line[10:15] # WBAN
            # line[28:34] # latitude
            # line[34:41] # longitude
            # line[46:51] # elevation
            # line[92:93] # temperature reading quality
            # year = line[15:19]
            # month = line[19:21]
            # day = line[21:23]
            # hour = line[23:25]
            # minute = line[25:27]
            air_temperature = Q_(float(line[87:92]) / 10, self._source_unit)
            if line[87:92] == "+9999":
                air_temperature = Q_(float("nan"),self._source_unit)
            self._data[line[15:25]] = air_temperature

class TMY3WeatherSource(WeatherSourceBase):
    def __init__(self,station_id):
        self.station_id = station_id
        self._data = {}
        self._source_unit = ureg.degC
        r = requests.get("http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/{}TYA.CSV".format(station_id))

        for line in r.text.splitlines()[3:]:
            row = line.split(",")
            date_string = row[0][0:2] + row[0][3:5] + row[1][0:2] # MMDDHH
            self._data[date_string] = Q_(float(row[31]),self._source_unit)

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        null = Q_(float("nan"),self._source_unit)
        day_str = day.strftime("%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self._data.get("{}{:02d}".format(day_str,i),null).to(unit).magnitude
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def annual_daily_temperatures(self,unit):
        """Returns a list of daily temperature normals for a typical
        meteorological year.
        """
        null = Q_(float("nan"),self._source_unit)
        start_day = datetime(2012,1,1)
        temps = []
        for days in range(365):
            day = start_day + timedelta(days=days)
            day_temps = []
            for hour in range(24):
                time = day + timedelta(seconds=hour*3600)
                temp = self._data.get(time.strftime("%m%d%H"),null).to(unit).magnitude
                day_temps.append(temp)
            day_data = np.array(day_temps)
            masked_data = np.ma.masked_array(day_data,np.isnan(day_data))
            # wrap in array for compatibility with model input format
            temps.append([np.mean(masked_data)])
        return temps

class WeatherUndergroundWeatherSource(WeatherSourceBase):
    def __init__(self,zipcode,start,end,api_key):
        self._data = {}
        assert end >= start
        date_format = "%Y%m%d"
        date_range_limit = 32
        end_date_str = datetime.strftime(end, date_format)
        for days in range(0, (end - start).days, date_range_limit):
            start_date = start + timedelta(days=days)
            start_date_str = datetime.strftime(start_date, date_format)
            query = 'http://api.wunderground.com/api/{}/history_{}{}/q/{}.json'\
                    .format(api_key,start_date_str,end_date_str,zipcode)
            self._get_query_data(query)

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance.
        """
        return self._data[day.strftime("%Y%m%d")]["meantempi"].to(unit).magnitude

    def _get_query_data(self,query):
        unit = ureg.degF
        for day in requests.get(query).json()["history"]["dailysummary"]:
            date_string = day["date"]["year"] + day["date"]["mon"] + \
                    day["date"]["mday"]
            data = {"meantempi":Q_(int(day["meantempi"]),unit)}
            self._data[date_string] = data

def nrel_tmy3_station_from_lat_long(lat,lng,api_key):
    """Use the National Renewable Energy Lab API to find the closest weather
    station for a particular lat/long. Requires a (freely available) API key.
    """
    result = requests.get("http://developer.nrel.gov/api/solar/data_query/"
                          "v1.json?api_key={}&lat={}&lon={}".format(api_key,
                                                                    lat,lng))
    result_json = result.json()
    if result_json["errors"] == []:
        if result_json['outputs']['tmy3'] is None:
            return None
        return result_json['outputs']['tmy3']['id'][2:]
    else:
        raise ValueError(result_json["errors"][0])

def ziplocate_us(zipcode):
    """Use the Ziplocate API to find the population-weighted lat/long centroid
    for this ZIP code.
    """
    result = requests.get('http://ziplocate.us/api/v1/{}'.format(zipcode))
    if result.status_code == 200:
        data = result.json()
        return data.get('lat'), data.get('lng')
    elif result.status_code == 404:
        raise ValueError("No known lat/long centroid for this ZIP code.")
    else:
        return None

def usaf_station_from_zipcode(zipcode,nrel_api_key):
    """Return a station identifier given a zipcode.
    """
    lat,lng = ziplocate_us(zipcode)
    station = nrel_tmy3_station_from_lat_long(lat,lng,nrel_api_key)
    return station

def haversine(lat1,lng1,lat2,lng2):
    pass

def lat_lng_to_tmy3(lat,lng):
    pass

def lat_lng_to_zipcode(lat,lng):
    pass

def tmy3_to_lat_lng(station):
    pass

def tmy3_to_zipcode(station):
    pass

def zipcode_to_lat_lng(zipcode):
    pass

def zipcode_to_tmy3(zipcode):
    pass
