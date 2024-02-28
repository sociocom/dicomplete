# dicomplete 医療辞書生成システム

## 説明

入力項目（input_column）から予測項目（predict_column）を信頼度（rank）を基に生成するシステム．

rank を指定するとそれ以上の信頼度のデータを訓練に，それ未満の信頼度のデータをテストにできる．<br>
例）rank="A"<br>
信頼度 S, A を訓練データ，
信頼度 B, C, D, E をテストデータとする．

## コマンド一覧

help が見れるコマンド: `rye run python main.py --help`<br>
入力すると以下の説明が見られる．

```
NAME
main.py

SYNOPSIS
main.py FNAME <flags>

POSITIONAL ARGUMENTS
FNAME

FLAGS
-b, --batch_size=BATCH_SIZE
Default: 128
-s, --src_token_max_length=SRC_TOKEN_MAX_LENGTH
Default: 16
-t, --tgt_token_max_length=TGT_TOKEN_MAX_LENGTH
Default: 16
-e, --epoch_num=EPOCH_NUM
Default: 100
-d, --device=DEVICE
Default: 'cuda:0'
-i, --input_column=INPUT_COLUMN
Default: 'appearance'
-p, --predict_column=PREDICT_COLUMN
Default: 'appearance_yomi'
-m, --model_name=MODEL_NAME
Default: 'retrieva-jp/t5-base-long'
-r, --rank=RANK
Default: 'S'
```
