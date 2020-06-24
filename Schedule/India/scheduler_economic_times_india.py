import sys
import os
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../Loader/India'))
sys.path.append(os.path.abspath('../../Schedule'))
sys.path.append(os.path.abspath('../../Database'))
import economic_times as loader
import schedule
from datetime import datetime, timedelta
import calendar
import time
import calendar_schedule as cs
import Database.database_log as database_log
import Database.database as sql_execute

sys.path.append(os.path.abspath('../../'))
import app_config

calendar.setfirstweekday(calendar.SUNDAY)

now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
current_time = current_time + ' ' + str(now.second * 1000)


# India market
# Global variables
market_hours_init = 0
lookup_value = "economic_times"  # search website name
df_config = sql_execute.read_configuration(lookup_value)  # read config file setting

row_id = df_config.index[0]
scheduled_task_sleeping = df_config["scheduled_task_sleeping"][row_id]
time_zone = df_config["market_time_zone"][row_id]
market_start_time = df_config["market_start_time"][row_id]
market_end_time = df_config["market_end_time"][row_id]
market_hours_delay = df_config["market_hours_delay"][row_id]
market_off_hours_delay = int(df_config["market_off_hours_delay"][row_id])
market_weekend_hours_delay = int(df_config["market_weekend_hours_delay"][row_id])

# remove existing scheduler
schedule.clear()

# declare global scheduler job
job = schedule.jobs


def read_market_hours():
    market_hour = cs.scheduler_zone(time_zone, market_start_time, market_end_time)
    return int(market_hour)


def call_loader():
    loader.start_load_process(lookup_value)


def scheduler(market_hour):

    global market_hours_init
    global job

    if market_hour == 1:
        market_hours_init = market_hour
        job = schedule.every(market_hours_delay).minutes.do(call_loader)
    elif market_hour == 2:
        market_hours_init = market_hour
        job = schedule.every(market_off_hours_delay).minutes.do(call_loader)
    elif market_hour == 3:
        market_hours_init = market_hour
        schedule.every(market_weekend_hours_delay).minutes.do(call_loader)


def cancel_job():
    schedule.cancel_job(job)
    print(schedule.jobs)  # shows empty list, as they are no open jobs
    time.sleep(5)  # just to double check that it does no longer trigger


def start_scheduler():
    market_hour = read_market_hours()
    scheduler(market_hour)


def start_process():

    try:

        print("***********Initiate Process - economic_times**************")
        database_log.process_log("India - economic_times : start_process", "Initiate Process")

        while True:
            # Checks whether a scheduled task - is pending to run or not
            scheduled_sleeping_seconds = app_config.india_market_scheduled_task_sleeping
            loader.start_load_process(lookup_value)
            database_log.process_log("India - economic_times : start_process", "Re-Run Process")

            print("Last run was successful for India - economic_times, next run in {} seconds.".format(scheduled_sleeping_seconds))
            time.sleep(scheduled_sleeping_seconds)

    except Exception as error:
        database_log.error_log("India - economic_times : start_process", error)
