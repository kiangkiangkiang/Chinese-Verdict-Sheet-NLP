import typer
from typing import Any
from verdict_analysis.information_extraction.run_eval import main as main_eval

information_extraction_app = typer.Typer()


@information_extraction_app.command("eval")
def eval(args: Any):
    breakpoint()
    main_eval(parser=args)


def train_cmd(name: str):
    print(f"train CMD in information: {name}")
    # train.run(name)


@information_extraction_app.command("predict")
def predict_cmd(name: str):
    print(f"predict CMD in information: {name}")
    # predict.run(name)
