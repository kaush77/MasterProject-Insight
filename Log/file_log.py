from datetime import datetime

log_status_path = "../../Log/log_status.txt"
log_error_path = "../../Log/error_log.txt"
now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M_%S")


def log_status(source, message):
    with open(log_status_path, "a+") as f:
        f.write("{} - Source :{} - {} \n".format(current_time, source, message))


def log_error(source, message):
    with open(log_error_path, "a+") as f:
        f.write("{} - Source :{} - {} \n".format(current_time, source, message))


if __name__ == '__main__':
    log_status("test")

