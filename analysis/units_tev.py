"""Tick formatters that display GeV-valued data in TeV units."""
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator


def log_shift(axis, shift, numticks=40):
    """Log axis holding GeV^p data: label 10^n as 10^(n+shift) (shift = -3p)."""
    axis.set_major_locator(LogLocator(base=10, numticks=numticks))
    axis.set_major_formatter(FuncFormatter(lambda x, pos: rf'$10^{{{int(round(np.log10(x))) + shift}}}$' if x > 0 and abs(np.log10(x) - round(np.log10(x))) < 1e-6 else ''))


def lin_scale(axis, fac=1e-3, fmt='{:g}'):
    axis.set_major_formatter(FuncFormatter(lambda x, pos: fmt.format(round(x * fac, 6))))


def tex_tev4(x_gev4):
    m, e = f"{x_gev4 / 1e12:.1e}".split('e'); e = int(e)
    return rf"10^{{{e}}}" if m == '1.0' else rf"{m}\times10^{{{e}}}"
