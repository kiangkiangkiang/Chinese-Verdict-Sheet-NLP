from paddle.static import InputSpec
from paddlenlp.utils.log import logger
import yaml
import os

ENTITY_TYPE = ["精神慰撫金額", "醫療費用", "薪資收入"]

UIE_INPUT_SPEC = [
    InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
]

logger.set_level("INFO")

REGULARIZED_TOKEN = [r"\n", r" ", r"\u3000", r"\\n"]


def load_config(yaml_file):
    yaml_paht = os.path.join("./src/chinese_verdict_sheet_nlp/information_extraction/config/", yaml_file)
    with open(yaml_paht, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
