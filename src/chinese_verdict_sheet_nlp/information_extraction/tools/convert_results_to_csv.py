import json
import csv
import argparse
from typing import List

ENTITY_TYPE = ["精神慰撫金額", "醫療費用", "薪資收入"]
REMAIN_KEYS = ["text"]
COLUMN_NAME_OF_JSON_CONTENT = "jfull_compress"
KEYS_MAPPING_TO_CSV_TABLE = {
    "start": "uie_result_start_index",
    "end": "uie_result_end_index",
    "probability": "uie_result_probability",
}


def read_uie_inference_results(path: str) -> List[dict]:
    """Get the UIE results made by run_infer.py

    Args:
        path (str): Path of UIE results.

    Returns:
        _type_: List of UIE results.
    """
    uie_result_list = []
    with open(path, "r", encoding="utf8") as f:
        result_list = json.loads(f.read())
        uie_result_list = [result for result in result_list]
    return uie_result_list


# fill nan
def uie_result_fill_null_entity(uie_result, fill_text_when_null: str = "nan"):
    for entity in ENTITY_TYPE:
        if not uie_result[0].get(entity):
            uie_result[0].update({entity: [{"text": fill_text_when_null, "start": -1, "end": -1, "probability": 0.0}]})
    return uie_result


# max filter
def uie_result_max_select(uie_result):
    new_result = [{}]
    for entity in uie_result[0]:
        new_result[0][entity] = [sorted(uie_result[0][entity], key=lambda x: x["probability"], reverse=True)[0]]
    return new_result


# select key
def uie_result_key_remain(uie_result, remain_key_in_csv: List[str]):
    new_result = [{}]
    for entity in uie_result[0]:
        tmp_list = []
        for each_result_in_entity in uie_result[0][entity]:
            tmp_dict = {}
            for key in remain_key_in_csv:
                tmp_dict.update({key: each_result_in_entity[key]})
            tmp_list.append(tmp_dict)
        new_result[0][entity] = tmp_list
    return new_result


# only work in single result
def adjust_verdict_to_csv_format(
    verdict,
    remain_key_in_csv: List[str],
    drop_keys: List[str] = [COLUMN_NAME_OF_JSON_CONTENT, "InferenceResults"],
):
    update_entity_result = {}
    for entity in verdict["InferenceResults"][0]:
        for key in remain_key_in_csv:
            if key == "text":
                update_entity_result.update({entity: verdict["InferenceResults"][0][entity][0]["text"]})
            else:
                update_entity_result.update(
                    {f"{KEYS_MAPPING_TO_CSV_TABLE[key]}_for_{entity}": verdict["InferenceResults"][0][entity][0][key]}
                )

    for drop_key in drop_keys:
        verdict.pop(drop_key)

    verdict.update(update_entity_result)
    return verdict


def write_json_list_to_csv(file_list, write_keys=None, save_dir="./verdict8000_uie_inference_result.csv"):
    header = write_keys if write_keys else list(file_list[0].keys())
    with open(save_dir, "w", encoding="utf_8_sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for file in file_list:
            data = [file[key] for key in header]
            writer.writerow(data)


if __name__ == "__main__":
    """將「run_infer.py」inference 產生的結果，轉換成 csv 格式。

    Example:
        python src/chinese_verdict_sheet_nlp/information_extraction/tools/convert_results_to_csv.py \
            --uie_results_path ./reports/information_extraction/inference_results/inference_results.json 

    Raises:
        ValueError: uie_results_path is not found.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--uie_results_path", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="uie_result_for_csv.csv")
    args = parser.parse_args()

    uie_inference_results = read_uie_inference_results(path=args.uie_results_path)

    for i, inference_result in enumerate(uie_inference_results):
        uie_inference_results[i]["InferenceResults"] = uie_result_fill_null_entity(
            uie_result=inference_result["InferenceResults"]
        )

        uie_inference_results[i]["InferenceResults"] = uie_result_max_select(
            uie_result=inference_result["InferenceResults"]
        )

        uie_inference_results[i]["InferenceResults"] = uie_result_key_remain(
            uie_result=inference_result["InferenceResults"], remain_key_in_csv=REMAIN_KEYS
        )

        uie_inference_results[i] = adjust_verdict_to_csv_format(inference_result, remain_key_in_csv=REMAIN_KEYS)

    write_json_list_to_csv(uie_inference_results)
