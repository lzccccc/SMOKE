import functools
import logging
import os
import sys


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output_dir, distributed_rank=0, name="smoke", file_name="log.txt"):
    '''
    Args:
        output_dir (str): a directory saves output log files
        name (str): name of the logger
        file_name (str): name of log file
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_dir:
        fh = logging.FileHandler(os.path.join(output_dir, file_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
