from contextlib import redirect_stdout, nullcontext
import os

def quiet_zone(verbose):
    """Возвращает контекстный менеджер в зависимости от verbose"""
    if verbose == 0:
        return redirect_stdout(open(os.devnull, 'w'))
    else:
        return nullcontext()