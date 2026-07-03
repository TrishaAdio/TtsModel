"""
console.py
----------
Small colorama-based helpers for consistent, colored CLI output across the
pipeline scripts. Import these instead of calling print() directly.

    from console import info, ok, warn, err, step, banner, progress_bar
"""
from __future__ import annotations

import shutil
import sys

from colorama import Fore, Style, init as _cinit

# autoreset=True so every print resets color automatically.
_cinit(autoreset=True)


def banner(title: str) -> None:
    width = min(shutil.get_terminal_size((80, 20)).columns, 70)
    line = "=" * width
    print(Fore.MAGENTA + Style.BRIGHT + line)
    print(Fore.MAGENTA + Style.BRIGHT + title.center(width))
    print(Fore.MAGENTA + Style.BRIGHT + line)


def step(n: int, total: int, msg: str) -> None:
    print(Fore.CYAN + Style.BRIGHT + f"[{n}/{total}] " + Style.RESET_ALL + Fore.CYAN + msg)


def info(msg: str) -> None:
    print(Fore.BLUE + "  i " + Style.RESET_ALL + msg)


def ok(msg: str) -> None:
    print(Fore.GREEN + Style.BRIGHT + "  \u2713 " + Style.RESET_ALL + Fore.GREEN + msg)


def warn(msg: str) -> None:
    print(Fore.YELLOW + Style.BRIGHT + "  ! " + Style.RESET_ALL + Fore.YELLOW + msg)


def err(msg: str) -> None:
    print(Fore.RED + Style.BRIGHT + "  \u2717 " + Style.RESET_ALL + Fore.RED + msg, file=sys.stderr)


def value(label: str, val) -> None:
    print("    " + Fore.WHITE + Style.DIM + f"{label}: " + Style.RESET_ALL + Fore.WHITE + str(val))


def progress_bar(current: int, total: int, prefix: str = "", width: int = 30) -> None:
    """In-place progress bar. Call with current=total at the end to finalize."""
    total = max(total, 1)
    filled = int(width * current / total)
    bar = Fore.GREEN + "\u2588" * filled + Style.RESET_ALL + Style.DIM + "\u2500" * (width - filled) + Style.RESET_ALL
    pct = int(100 * current / total)
    end = "\n" if current >= total else ""
    sys.stdout.write(f"\r  {prefix} [{bar}] {pct:3d}% ({current}/{total}){end}")
    sys.stdout.flush()
