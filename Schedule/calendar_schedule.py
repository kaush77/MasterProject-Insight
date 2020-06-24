from datetime import date,datetime
import calendar
import pytz
import pandas as pd

# global variables
# config file path
config_path = "../config/config.csv"
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']


def weekdays():
    return weekdays;

def current_day(tZone):
    tz = pytz.timezone(tZone)
    current_datetime = datetime.now(tz)
    return current_datetime.strftime("%A")


def is_weekend(tZone):
    currentDay = current_day(tZone)
    if currentDay in ['Saturday', 'Sunday']:
        return True
    else:
        return False


def current_time(tZone):
    tz = pytz.timezone(tZone)
    current_datetime = datetime.now(tz)
    return current_datetime.strftime("%H"), current_datetime.strftime("%M")


def market_time(trading_hours):
    trading_hours = trading_hours.split("_")
    return trading_hours[0],trading_hours[1]


# 1 - trading_hour, 2 - trading_off_hour 3 - trading_weekend_hour
def scheduler_zone(time_zone, market_start_time, market_end_time):
    if not is_weekend(time_zone):
        zone_hours, zone_minutes = current_time(time_zone)
        market_start_hours, market_start_minutes = market_time(market_start_time)
        market_end_hours, market_end_minutes = market_time(market_end_time)

        # print("{}-{}".format(zone_hours, zone_minutes))
        if (int(zone_hours) >= int(market_start_hours) and int(zone_minutes) >= int(market_start_minutes)
                and (int(zone_hours) <= int(market_end_hours) and int(zone_minutes) <= int(market_end_minutes))):
            return 1
        else:
            return 2
    else:
        return 3


if __name__ == '__main__':
    print(is_weekend("America/Vancouver"))
