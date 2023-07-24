"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mchinese_verdict_sheet_nlp` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``chinese_verdict_sheet_nlp.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``chinese_verdict_sheet_nlp.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import argparse
from chinese_verdict_sheet_nlp.information_extraction.run_eval import main as main_eval
from chinese_verdict_sheet_nlp.information_extraction.run_convert import main as main_convert
from chinese_verdict_sheet_nlp.information_extraction.run_train import main as main_train


def main(args=None):
    parser = argparse.ArgumentParser(description="Command description.")
    parser.add_argument("clsorie", type=str)
    parser.add_argument("dowhat", type=str)
    args = parser.parse_known_args()
    if args[0].dowhat == "eval":
        main_eval()
    if args[0].dowhat == "convert":
        main_convert()
    if args[0].dowhat == "train":
        main_train()
