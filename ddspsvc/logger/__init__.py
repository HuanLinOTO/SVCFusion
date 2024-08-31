from loguru import logger

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress as _Progress
from rich.progress import (
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
)
from rich.text import Text

console = Console(stderr=None)


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


class SpeedColumn(ProgressColumn):
    def render(self, task: any) -> any:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed}/s", style="progress.data.speed")


def Progress():
    return _Progress(
        TextColumn("[progress.description]{task.description}"),
        # TextColumn("[progress.description]W"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        # TextColumn("[progress.data.speed]{task.speed:06.2f}"),
        SpeedColumn(),
        TextColumn("[red]*Elapsed[/red]"),
        TimeElapsedColumn(),
        console=console,
    )
