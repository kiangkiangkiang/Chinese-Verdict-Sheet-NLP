# 判決書 - 金額擷取模型

--- 
# Environment
## Required

- python >= 3.7
- paddlenlp >= 2.5.2
- paddlepaddle-gpu >= 2.5.0 (**單卡**情況下使用 GPU) 
## Optional
### 使用 CPU 或 多卡GPU 請安裝以下套件

- paddlepaddle-gpu == 2.4.2.post116 (**多卡**情況下使用 GPU)
- paddlepaddle == 2.4.2 (使用 CPU)

### Inference 時，使用 Taskflow 時，若精度為 `fp16` 請安裝以下套件

- onnx >= 1.14.0
- onnxconverter-common >= 1.13.0
- onnxruntime-gpu >= 1.14.1

### 使用 convert_csv_money.py 轉換金錢格式，請安裝以下套件

- cn2an >= 0.5.20
- OpenCC >= 1.1.6

# Quick Start

所有參數檔案請從 `.src/chinese_verdict_sheet_nlp/information_extraction/config` 更改。
## 0. 設定環境變數

至 Chinese-Verdict-Sheet-NLP 資料夾底下輸入：
`export PYTHONPATH="$PWD/src"`
## 1. Convert Function

將 label studio 針對 UIE 任務所標記完的資料匯出後，透過 run_convert.py 轉換成模型所吃的 .txt 檔案，並依照比例切割成訓練資料集、驗證資料集、測試資料集後匯出。

``` python
python -m chinese_verdict_sheet_nlp information_extraction convert
```
### 重要參數

- `labelstudio_file`: 預設`./data/information_extraction/label_studio_data/label_studio_output.json`，label studio 標記完後匯出的 JSON 檔案。
- `save_dir`: 預設`./data/model_input_data/`，轉換後的 txt 檔案。
- `split_ratio`: 預設`[0.8, 0.1, 0.1]`，訓練資料集、驗證資料集、測試資料集各個佔比。
- `is_regularize_data`: 預設`True`，是否在轉換前清除特殊字元，ex. "\n"。
## 2. Training Function

微調模型的主要運行程式。
### 單卡訓練

``` python
python -m chinese_verdict_sheet_nlp information_extraction train
```
### 多卡訓練

使用 `--gpus` 指定顯卡（請確認 `paddlepaddle-gpu` 版本無誤）。

``` python
python -u -m paddle.distributed.launch --gpus "0,1,2,3"  -m src/chinese_verdict_sheet_nlp information_extraction train
```

### 重要參數

- `device`: 預設`gpu`，選擇用何種裝置訓練模型，可使用`cpu`或是指定 gpu ，例如：`gpu:0`。
- `model_name_or_path`: 預設`uie-base`，訓練時所使用的模型或是模型 checkpoint 路徑。
- `max_seq_len`: 預設`768`，模型在每個 batch 所吃的最大文本長度。
- `per_device_train_batch_size`: 預設`8`，模型在每個裝置訓練所使用的批次資料數量。
- `per_device_eval_batch_size`: 預設`8`，模型在每個裝置驗證所使用的批次資料數量。
- `dataset_path`: 預設`./data/model_input_data/`，主要存放資料集的位置。
- `train_file`: 預設`train.txt`，訓練資料集檔名。
- `dev_file`: 預設`dev.txt`，驗證資料集檔名。
- `test_file`: 預設`test.txt`，測試資料集檔名。
- `eval_steps`: 預設與`--logging_steps`相同，指模型在每幾個訓練步驟時要做驗證。
- `output_dir`: 模型訓練產生的 checkpoint 檔案位置。
- `metric_for_best_model`: 預設`loss`，訓練過程中，選擇最好模型的依據。

## Evaluation Function

驗證的主要運行程式。

``` python
python -m chinese_verdict_sheet_nlp information_extraction eval
```

### 重要參數

- `device`: 預設`gpu`，選擇用何種裝置訓練模型，可使用`cpu`或是指定 gpu ，例如：`gpu:0`。
- `model_name_or_path`: 預設`uie-base`，訓練時所使用的模型或是模型 checkpoint 路徑。
- `max_seq_len`: 預設`768`，模型在每個 batch 所吃的最大文本長度。
- `dev_file`: 預設`./data/model_input_data/test.txt`，驗證資料集的檔案路徑。
- `batch_size`: 預設`8`，模型所使用的批次資料數量。
- `is_eval_by_class`: 預設`False`，是否根據不同類別算出各自指標。

## Inference Function

預測的主要運行程式。

``` python
python -m chinese_verdict_sheet_nlp information_extraction infer
```

### 重要參數

- `data_file`: 預設`dev.txt`，驗證資料集檔名。
- `save_dir`: 模型訓練產生的 checkpoint 檔案位置。
- `is_regularize_data`: 預設`False`，是否在轉換前清除特殊字元，ex. "\n"。
- `precision`: 預設`fp32`，模型推論時的精確度，可使用`fp16` (only for gpu) 或`fp32`，其中`fp16`較快，使用`fp16`需注意CUDA>=11.2，cuDNN>=8.1.1，初次使用需按照提示安装相關依賴（`pip install onnxruntime-gpu onnx onnxconverter-common`）。
- `batch_size`: 預設`1`，模型所使用的批次資料數量。
- `taskpath`: 用來推論所使用的 checkpoint 檔案位置。
- `select_strategy`: 預設`all`，模型推論完後，保留推論結果的策略，`all`表示所有推論結果皆保留。其他可選`max`，表示保留機率最高的推論結果。`threshold`表示推論結果機率值高於`select_strategy_threshold`的結果皆保留。
- `select_strategy_threshold`: 預設`0.5`，表示當`select_strategy=threshold`時的門檻值。
- `select_key`: 預設`text start end probability`，表示最終推論保留的值。僅保留文字及機率可設`text probability`。




