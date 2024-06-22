import click
import richuru
from loguru import logger

from cli.resample import resample
from cli.slice_audio import slice_audio, slice_audio_v2
from cli.convert_to_wav import to_wav


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool):
    """An audio preprocessing CLI."""

    if debug:
        richuru.install()
        logger.info("Debug mode is on")


# Register subcommands
cli.add_command(slice_audio)
cli.add_command(slice_audio_v2)
cli.add_command(resample)
cli.add_command(to_wav)
if __name__ == "__main__":
    cli()
