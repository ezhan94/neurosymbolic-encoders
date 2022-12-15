import logging
import os
from .. import dsl


def init_logging(save_path):
    logfile = os.path.join(save_path, 'log.txt')

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO)

def log_and_print(line):
    print(line)
    logging.info(line)

def print_program(program, ignore_constants=True):
    print("WARNING: print_program is deprecated, use program.to_str()")
    return program.to_str(include_params=(not ignore_constants))

def print_program_dict(prog_dict):
    log_and_print(prog_dict["program"].to_str())
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))
