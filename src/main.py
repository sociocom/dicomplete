import pandas as pd
import torch
import fire
from generate import Trainer, GenerateText
from transformers import T5Tokenizer
import pandas as pd
import torch
import fire
from generate import Trainer, GenerateText
from transformers import T5Tokenizer


def main(
    fname,
    batch_size=128,
    src_token_max_length=16,
    tgt_token_max_length=16,
    epoch_num=100,
    device="cuda:0" if torch.cuda.is_available() else "mps",
    input_column="appearance",
    predict_column="appearance_yomi",
    model_name="retrieva-jp/t5-base-long",
    rank="S",
):
    print(
        f" input_column: {input_column}\n target_column: {predict_column}\n rank: {rank}\n batch_size: {batch_size}\n token_max_length_src: {src_token_max_length}\n token_max_length_tgt: {tgt_token_max_length}\n epoch_num: {epoch_num}\n device: {device}\n model_name: {model_name}\n"
    )
    # データの読み込み
    # df = pd.read_csv("./data/db_data_DISEASE_SIP-3_v202401_2.1.csv")
    df = pd.read_csv(fname)
    print(df.head())

    # rank <- A
    ranks = "SABCDE"
    ix_train = ranks.index(rank) + 1  # => 1 + 1 = 2
    ranks_train = ranks[:ix_train]  # => "SA"
    str_ranks_train = r"|".join(r for r in ranks_train)  # => "S|A"
    mask_train = (
        df[f"{predict_column}_reliability"].fillna("E").str.contains(str_ranks_train)
    )
    df_train = df[mask_train][
        [input_column, predict_column, f"{predict_column}_reliability"]
    ].head(1000)
    df_test = df[~mask_train][
        [input_column, predict_column, f"{predict_column}_reliability"]
    ].head(
        1000
    )  # ~mask_train == not mask_train

    print("df_train")
    print(df_train.head())
    print("df_test")
    print(df_test.head())

    # 事前学習済みモデル
    MODEL_NAME = model_name
    # トークナイザー（SentencePiece）モデルの読み込み
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_column = input_column
    predict_column = predict_column

    # Trainerの初期化とトレーニング
    trainer = Trainer(
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        train_data=df_train,
        input_column=input_column,
        target_column=predict_column,
        batch_size=batch_size,
        token_max_length_src=src_token_max_length,
        token_max_length_tgt=tgt_token_max_length,
        epochs=epoch_num,
        device=device,
    )
    best_model, best_tokenizer, timestamp = trainer.train()

    # テキスト生成の準備
    text_generator = GenerateText(
        best_model,
        best_tokenizer,
        df_test,
        input_column,
        predict_column,
        timestamp,
        batch_size=batch_size,
        token_max_length_src=src_token_max_length,
        token_max_length_tgt=tgt_token_max_length,
        device=device,
    )

    # テキスト生成
    generated_df = text_generator.generate_text()
    print(generated_df.head())


if __name__ == "__main__":
    # main()
    fire.Fire(main)
