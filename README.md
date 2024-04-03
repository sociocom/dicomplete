# dicomplete 医療辞書生成システム

## 説明

入力項目（input_column）から予測項目（predict_column）を信頼度（rank）を基に生成するシステム．

rank を指定するとそれ以上の信頼度のデータを訓練に，それ未満の信頼度のデータをテストにできる．<br>
例）rank="A"<br>
信頼度 S, A を訓練データ，
信頼度 B, C, D, E をテストデータとする．

## 使用方法

Rye を導入した後，`rye sync`で python ライブラリをインストール

下記のコマンド一覧を参考に適切な引数を追加し，実行．<br>
実行例

```
rye run python src/main.py -i='正規形' -p='正規形よみ' --reliability_column='flag' 'data/inputs/db.csv'
```

## コマンド一覧

help が見れるコマンド: `rye run python src/main.py --help`<br>
入力すると以下の説明が見られる．

```
NAME
    main.py

SYNOPSIS
    main.py FNAME <flags>

POSITIONAL ARGUMENTS
    FNAME
        Type: str

FLAGS
    -b, --batch_size=BATCH_SIZE
        Type: int
        Default: 8
    -s, --src_token_max_length=SRC_TOKEN_MAX_LENGTH
        Type: int
        Default: 16
    -t, --tgt_token_max_length=TGT_TOKEN_MAX_LENGTH
        Type: int
        Default: 16
    -e, --epoch_num=EPOCH_NUM
        Type: int
        Default: 10
    -d, --device=DEVICE
        Type: str
        Default: 'cuda:0'
    -i, --input_column=INPUT_COLUMN
        Type: str
        Default: 'appearance'
    -p, --predict_column=PREDICT_COLUMN
        Type: str
        Default: 'appearance_yomi'
    --reliability_column=RELIABILITY_COLUMN
        Type: str
        Default: 'appearance_reliability'
    -m, --model_name=MODEL_NAME
        Type: str
        Default: 'retrieva-jp/t5-base-long'
    --rank=RANK
```
