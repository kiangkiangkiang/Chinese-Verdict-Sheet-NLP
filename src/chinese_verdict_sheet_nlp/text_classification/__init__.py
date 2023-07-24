import typer

text_classification_app = typer.Typer()


@text_classification_app.command("train")
def test_train(args: str = "config_path.yaml"):
    print("in __init__ of text_classification")
    print(f"args={args}")
