import datetime
import os
import time

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as _Progress

console = Console(stderr=None)

logger.remove()


def format_level(str, length):
    if len(str) < length:
        str = str + " " * (length - len(str))
    else:
        str = str
    # 给 str 上对应 level 的颜色
    if str == "INFO   ":
        str = f"[bold green]{str}[/bold green]"
    elif str == "WARNING":
        str = f"[bold yellow]{str}[/bold yellow]"
    elif str == "ERROR  ":
        str = f"[bold red]{str}[/bold red]"
    elif str == "DEBUG  ":
        str = f"[bold cyan]{str}[/bold cyan]"
    return str


def default_format(record):
    return f"[green]{record['time'].strftime('%Y-%m-%d %H:%M:%S')}[/green] | [level]{format_level(record['level'].name,7)}[/level] | [cyan]{record['file'].path.replace(os.getcwd()+os.sep,'')}:{record['line']}[/cyan] - [level]{record['message']}[/level]\n"


logger.add(lambda m: console.print(m, end=""), format=default_format, colorize=True)


def addLogger(path):
    logger.add(
        path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}",
        colorize=True,
    )


info = logger.info
error = logger.error
warning = logger.warning
warn = logger.warning
debug = logger.debug


def hps(hps):
    console.print(hps)


use_gradio_progress = False


class GradioProgress:
    def __init__(self) -> None:
        import gradio as gr

        self.progress = gr.Progress

    def __getattr__(self, name):
        return getattr(self.progress, name)

    def __enter__(self):
        return self

    def __exit__(self, *args): ...


class ProgressProxy:
    def __init__(self, progress) -> None:
        self.progress = progress

    def __getattr__(self, name):
        global use_gradio_progress, info
        # info(name)
        if name == "track" and use_gradio_progress:
            import gradio as gr

            return gr.Progress().tqdm

        res = getattr(self.progress, name)
        return res

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        self.progress.__exit__(*args)


def Progress():  # noqa: F811
    return ProgressProxy(
        _Progress(
            TextColumn("[progress.description]{task.description}"),
            # TextColumn("[progress.description]W"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[red]*Elapsed[/red]"),
            TimeElapsedColumn(),
            console=console,
        )
    )
