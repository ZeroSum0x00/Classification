import logging



def get_logger(name="CLS"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Tạo formatter với mã màu ANSI cho mỗi mức độ log
    _formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Tạo handler cho console
    console_handler = logging.StreamHandler()

    # Mức độ log cho console
    console_handler.setLevel(logging.DEBUG)
    
    # Cập nhật formatter với màu cho các mức độ log
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[94m",  # Màu xanh cho DEBUG
            "INFO": "\033[92m",   # Màu xanh lá cho INFO
            "WARNING": "\033[93m", # Màu vàng cho WARNING
            "ERROR": "\033[91m",  # Màu đỏ cho ERROR
            "CRITICAL": "\033[95m" # Màu tím cho CRITICAL
        }

        RESET = "\033[0m"

        def format(self, record):
            levelname = record.levelname
            log_message = super().format(record)
            color = self.COLORS.get(str(levelname), "")
            log_message = str(log_message)
            reset = self.RESET if hasattr(self, "RESET") else ""
            return f"{color}{log_message}{reset}"

    # Áp dụng ColoredFormatter cho handler
    colored_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(colored_formatter)

    # Nếu chưa có handler, thêm handler mới vào logger
    if len(logger.handlers) == 0:
        logger.addHandler(console_handler)

    return logger

# Tạo logger
logger = get_logger()
