import logging

def setup_logger(name: str, logfile: str = "app.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(logfile, encoding='utf-8'),   # goes to file
            logging.StreamHandler() # goes to console
        ]
    )
    return logging.getLogger(name)
