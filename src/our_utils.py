# Если этот файл переименовать в utils.py будет ошибка!
# Файл с именем utils.py уже существует в ./baselines/PRGPT

from contextlib import redirect_stdout, nullcontext, contextmanager
import warnings
import os

def print_zone(flag):
    if flag:
        return nullcontext()
    else:
        return redirect_stdout(open(os.devnull, 'w'))

@contextmanager
def suppress_warnings_context():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield