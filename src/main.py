import pandas as pd
import torch
import fire
from generate import Trainer, GenerateText
from transformers import T5Tokenizer


def main(
    fname: str,
    batch_size: int = 8,
    src_token_max_length: int = 16,
    tgt_token_max_length: int = 16,
    epoch_num: int = 10,
    device: str = "cuda:0" if torch.cuda.is_available() else "mps",
    input_column: str = "appearance",
    predict_column: str = "appearance_yomi",
    reliability_column: str = "appearance_reliability",
    model_name: str = "retrieva-jp/t5-base-long",
    rank: str = "S",
):
    print(
        f" input_column: {input_column}\n target_column: {predict_column}\n reliability_column: {reliability_column}\n rank: {rank}\n batch_size: {batch_size}\n token_max_length_src: {src_token_max_length}\n token_max_length_tgt: {tgt_token_max_length}\n epoch_num: {epoch_num}\n device: {device}\n model_name: {model_name}\n"
    )
    # データの読み込み
    df = pd.read_csv(fname)
    print(df.head())

    df_skipped = df[df[f"{reliability_column}"] != "E"]
    # rank <- A
    ranks = "SABCDE"
    ix_train = ranks.index(rank) + 1  # => 1 + 1 = 2
    ranks_train = ranks[:ix_train]  # => "SA"
    str_ranks_train = r"|".join(r for r in ranks_train)  # => "S|A"
    mask_train = (
        df_skipped[f"{reliability_column}"].fillna("D").str.contains(str_ranks_train)
    )
    df_train = df_skipped[mask_train][
        [input_column, predict_column, f"{reliability_column}"]
    ].reset_index(drop=True)
    df_test = df_skipped[~mask_train][
        ["ID", input_column, predict_column, f"{reliability_column}"]
    ].reset_index(
        drop=True
    )  # ~mask_train == not mask_train
    df_test["input_original"] = df_test[input_column]
    df_train[input_column] = df_train[input_column].fillna("").astype(str)
    df_train[predict_column] = df_train[predict_column].fillna("").astype(str)
    df_test[input_column] = df_test[input_column].fillna("").astype(str)
    df_test[predict_column] = df_test[predict_column].fillna("").astype(str)

    print("df_train")
    print(df_train.head())
    print("df_test")
    print(df_test.head())

    torch.manual_seed(2023)

    # 事前学習済みモデル
    MODEL_NAME = model_name
    # トークナイザー（SentencePiece）モデルの読み込み
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    best_model, best_tokenizer, timestamp, stop_epoch = trainer.train()

    # テキスト生成の準備
    text_generator = GenerateText(
        best_model,
        best_tokenizer,
        df,
        df_test,
        input_column,
        predict_column,
        reliability_column,
        timestamp,
        batch_size=batch_size,
        token_max_length_src=src_token_max_length,
        token_max_length_tgt=tgt_token_max_length,
        epoch=stop_epoch,
        device=device,
    )

    # テキスト生成
    generated_df = text_generator.generate_text()
    print(generated_df.head())


if __name__ == "__main__":
    fire.Fire(main)
