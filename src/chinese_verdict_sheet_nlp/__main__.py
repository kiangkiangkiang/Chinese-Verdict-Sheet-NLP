"""
Entrypoint module, in case you use `python -mchinese_verdict_sheet_nlp`.


Why does this file exist, and why __main__? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/2/using/cmdline.html#cmdoption-m
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""

from .information_extraction import information_extraction_app
from .text_classification import text_classification_app
import typer

if __name__ == "__main__":
    task = typer.Typer()
    task.add_typer(information_extraction_app, name="information_extraction")
    task.add_typer(text_classification_app, name="text_classification")
    task()
