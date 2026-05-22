from colorama import Fore, Style, init as colorama_init
import torch.distributed as dist
from typing import Any, Optional
import os

def colored_exception(exc_class, message):
	"""Raise an exception with colored message."""
	raise exc_class(Fore.RED + str(message) + Style.RESET_ALL)

colorama_init()


__all__ = [
    "colored_exception",
    "colorize",
    "print_warning",
    "print_info",
    "rank_zero_print_warning",
    "rank_zero_print_info",
    "rank_zero_print_error",
    "local_rank_zero_print_warning",
    "local_rank_zero_print_info",
    "_log_line",
]

_COLOR_MAP = {
    "yellow": Fore.YELLOW,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "blue": Fore.BLUE,
    "cyan": Fore.CYAN,
    "magenta": Fore.MAGENTA,
    "white": Fore.WHITE,
}

_DISABLE_COLOR = bool(os.environ.get("NO_COLOR"))


def colorize(text: str, color: str = "yellow") -> str:
    if _DISABLE_COLOR:
        return text
    prefix = _COLOR_MAP.get(color.lower())
    if not prefix:
        return text
    return f"{Style.BRIGHT}{prefix}{text}{Style.RESET_ALL}"


def print_warning(message: str, color: str = "yellow") -> None:
    print(colorize(f"[Warning] {message}", color), flush=True)


def print_info(message: str, color: str = "cyan") -> None:
    print(colorize(f"[Info] {message}", color), flush=True)


def _is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    for key in ("RANK", "LOCAL_RANK", "PMI_RANK"):
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            return int(val) == 0
        except Exception:
            pass

    return True

def _is_local_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() % dist.get_local_world_size() == 0

    for key in ("LOCAL_RANK",):
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            return int(val) == 0
        except Exception:
            pass

    return True

def rank_zero_print_warning(message: str, color: str = "yellow") -> None:
	if _is_rank_zero():
		print_warning(message, color)
  
def local_rank_zero_print_warning(message: str, color: str = "yellow") -> None:
    if _is_local_rank_zero():
        print_warning(message, color)
        
def local_rank_zero_print_info(message: str, color: str = "cyan") -> None:
    if _is_local_rank_zero():
        print_info(message, color)


def rank_zero_print_info(message: str, color: str = "cyan") -> None:
	if _is_rank_zero():
		print_info(message, color)

def rank_zero_print_error(message: str, color: str = "red") -> None:
    if _is_rank_zero():
        print(colorize(f"[Error] {message}", color), flush=True)
        

def _log_line(message: str, *, progress_bar: Optional[Any] = None) -> None:
    """Write a log line without breaking active tqdm progress bars."""
    if progress_bar is not None:
        progress_bar.write(message)
    else:
        print(message)
