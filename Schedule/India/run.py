import multiprocessing
import sys
import os
sys.path.append(os.path.abspath('../../'))
import Database.database_log

# Schedule to run Indian stock news market
import scheduler_mint_india as mint
import scheduler_business_standard_india as business_standard
import scheduler_money_control_india as money_control
import scheduler_economic_times_india as economic_times
import scheduler_india_reuters_india as reuters
import scheduler_india_infoline_india as infoline
import scheduler_financial_express_india as financial_express
import scheduler_bloombergquint_india as bloombergquint
import scheduler_businesstoday_india as businesstoday



def run_news_scraper():
    try:

        with multiprocessing.Manager() as manager:

            # creating new processes
            p1 = multiprocessing.Process(target=mint.start_process)
            p2 = multiprocessing.Process(target=business_standard.start_process)
            p3 = multiprocessing.Process(target=money_control.start_process)
            p4 = multiprocessing.Process(target=economic_times.start_process)
            p5 = multiprocessing.Process(target=reuters.start_process)
            p6 = multiprocessing.Process(target=infoline.start_process)
            p7 = multiprocessing.Process(target=financial_express.start_process)
            p8 = multiprocessing.Process(target=bloombergquint.start_process)
            p9 = multiprocessing.Process(target=businesstoday.start_process)

            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()
            p8.start()
            p9.start()

    except Exception as error:
        database_log.error_log("read_website_configuration", error)


if __name__ == '__main__':

    try:

        run_news_scraper()

    except Exception as error:
        database_log.error_log("read_website_configuration", error)
