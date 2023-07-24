import typer
from .run_convert import main as main_convert
from .run_eval import main as main_eval
from .run_train import main as main_train
from .run_infer import main as main_infer
import os

information_extraction_app = typer.Typer()
config_base_path = "./src/chinese_verdict_sheet_nlp/information_extraction/config/"


@information_extraction_app.command("convert")
def convert(args: str = os.path.join(config_base_path, "convert_config.yaml")):
    main_convert(config_file=args)


@information_extraction_app.command("eval")
def eval(args: str = os.path.join(config_base_path, "eval_config.yaml")):
    main_eval(config_file=args)


@information_extraction_app.command("train")
def train(args: str = os.path.join(config_base_path, "train_config.yaml")):
    main_train(config_file=args)


@information_extraction_app.command("infer")
def infer(args: str = os.path.join(config_base_path, "infer_config.yaml")):
    main_infer(config_file=args)
