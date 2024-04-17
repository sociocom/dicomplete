import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from utils import T5FineTuner, generate_text_from_model

from tqdm import tqdm
from datetime import datetime
import random
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class CustomDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        input_column,
        target_column,
        max_source_length=8,
        max_target_length=8,
    ):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.input_column = input_column
        self.target_column = target_column
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert idx to integer if it's a string
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                raise ValueError(f"Invalid index value: {idx}")

        source_text = str(self.data.iloc[idx][self.input_column])
        target_text = str(self.data.iloc[idx][self.target_column])
        source = self.tokenizer.encode_plus(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "source_ids": source["input_ids"].flatten(),
            "source_mask": source["attention_mask"].flatten(),
            "target_ids": target["input_ids"].flatten(),
            "target_mask": target["attention_mask"].flatten(),
        }


class Trainer:
    def __init__(
        self,
        model_name,
        tokenizer,
        train_data,
        input_column,
        target_column,
        batch_size=4,
        epochs=3,
        learning_rate=1e-4,
        test_size=0.1,
        token_max_length_tgt=8,
        token_max_length_src=8,
        device="cpu",
    ):
        self.device = device
        self.model_name = model_name
        t5_fine_tuner = T5FineTuner(pretrained_model_name=model_name)
        self.model = t5_fine_tuner.to(self.device)
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.input_column = input_column
        self.target_column = target_column
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.max_length_tgt = token_max_length_tgt
        self.max_length_src = token_max_length_src
        self.patience = 5  # 早期停止の我慢できるエポック数
        self.min_val_loss = float("inf")  # 最小検証損失を保持する変数
        self.num_epochs_without_improvement = (
            0  # 検証損失が改善されないエポック数をカウントする変数
        )

    def train(self):
        best_loss = float("inf")
        best_model_state = None
        best_tokenizer_state = None
        best_accuracy = 0.0
        # testsplitに変更に9:1　user指定でもいいかも
        train_df, val_df = train_test_split(
            self.train_data,
            test_size=self.test_size,
            random_state=42,
            shuffle=True,
        )
        train_dataset = CustomDataset(
            train_df,
            tokenizer=self.tokenizer,
            input_column=self.input_column,
            target_column=self.target_column,
        )
        val_dataset = CustomDataset(
            val_df,
            tokenizer=self.tokenizer,
            input_column=self.input_column,
            target_column=self.target_column,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["source_ids"].to(self.device)
                attention_mask = batch["source_mask"].to(self.device)
                labels = batch["target_ids"].to(self.device)
                labels_attention_mask = batch["target_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=labels_attention_mask,
                )

                loss = outputs.loss
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss /= len(train_loader)

            val_loss, val_accuracy, all_targets, all_predictions = self.evaluate(
                val_df, val_loader, self.model, self.tokenizer
            )

            print(f"\nEpoch: {epoch+1}, Train Loss: {train_loss}")
            print(f"Val Accuracy: {val_accuracy}, Val Loss: {val_loss}")

            # 最小検証損失が更新された場合
            if val_loss < best_loss:
                best_loss = val_loss
                self.num_epochs_without_improvement = 0
                best_accuracy = val_accuracy
                best_model_state = copy.deepcopy(self.model)
                best_tokenizer_state = copy.deepcopy(self.tokenizer)
            else:
                self.num_epochs_without_improvement += 1
                # 早期停止条件を満たした場合
                if self.num_epochs_without_improvement >= self.patience:
                    print(
                        f"\nValidation loss has not improved for {self.patience} epochs. Training stopped."
                    )
                    print(f"Targets: {all_targets[:10]}")
                    print(f"Predicts: {all_predictions[:10]}")
                    break
            if ((epoch + 1) % 5 == 0) | (epoch + 1 == self.epochs):
                print(f"Targets: {all_targets[:10]}")
                print(f"Predicts: {all_predictions[:10]}")

        print(f"\nBest Accuracy: {best_accuracy}, Best Loss: {best_loss}")
        stop_epoch = epoch + 1
        # モデルの保存
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        best_model_state.model.save_pretrained(
            f"./models/{self.model_name}/{self.input_column}_to_{self.target_column}/epoch_{stop_epoch}/{timestamp}/model"
        )
        best_tokenizer_state.save_pretrained(
            f"./models/{self.model_name}/{self.input_column}_to_{self.target_column}/epoch_{stop_epoch}/{timestamp}/tokenizer"
        )

        return best_model_state, best_tokenizer_state, timestamp, stop_epoch

    def evaluate(self, dataframe, dataloader, model, tokenizer):
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["source_ids"].to(self.device)
                attention_mask = batch["source_mask"].to(self.device)
                labels = batch["target_ids"].to(self.device)
                labels_attention_mask = batch["target_mask"].to(self.device)

                # モデルに入力テンソルを渡して出力を取得
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=labels_attention_mask,
                )
                # 生成されたテキストのlossを計算
                loss = outputs.loss
                total_loss += loss.item()

            for i in range(0, len(dataframe), self.batch_size):
                batch_df = dataframe.iloc[i : i + self.batch_size, :]
                generated_text = generate_text_from_model(
                    tags=batch_df[self.input_column].to_list(),
                    trained_model=model,
                    tokenizer=tokenizer,
                    num_return_sequences=1,
                    max_length_src=self.max_length_src,
                    max_length_target=self.max_length_tgt,
                    num_beams=10,
                    device=self.device,
                )

                all_predictions.extend(generated_text)
        all_targets = dataframe[self.target_column].tolist()
        total_loss /= len(dataframe)

        val_loss = total_loss / len(dataframe)
        val_accuracy = accuracy_score(all_targets, all_predictions)

        return val_loss, val_accuracy, all_targets, all_predictions


class GenerateText:
    """GenerateText"""

    def __init__(
        self,
        model,
        tokenizer,
        df,
        val_dataset,
        input_column,
        target_column,
        reliability_column,
        timestamp,
        batch_size=16,
        token_max_length_src=8,
        token_max_length_tgt=8,
        epoch=10,
        device="cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.df = df
        self.input_column = input_column
        self.target_column = target_column
        self.reliability_column = reliability_column
        self.timestamp = timestamp
        self.batch_size = batch_size
        self.token_max_length_src = token_max_length_src
        self.token_max_length_tgt = token_max_length_tgt
        self.epoch = epoch
        self.device = device

    def generate_text(self):
        """Generate text"""
        original_texts = self.val_dataset["input_original"].to_list()
        target_columns_texts = self.val_dataset[self.target_column].to_list()
        reliability = self.val_dataset[self.reliability_column].to_list()
        prediction = []

        for i in tqdm(range(0, len(self.val_dataset), self.batch_size)):
            batch = self.val_dataset.iloc[i : i + self.batch_size, :]
            soap = batch[self.input_column].to_list()
            generated_text = generate_text_from_model(
                tags=soap,
                trained_model=self.model,
                tokenizer=self.tokenizer,
                num_return_sequences=1,
                max_length_src=self.token_max_length_src,
                max_length_target=self.token_max_length_tgt,
                num_beams=10,
                device=self.device,
            )
            prediction.extend(generated_text)

        column_df = pd.DataFrame(
            {
                "ID": self.val_dataset["ID"].to_list(),
                self.input_column: original_texts,
                self.target_column: target_columns_texts,
                f"{self.target_column}_generated": prediction,
                f"{self.reliability_column}": reliability,
            }
        )

        df = self.df[
            ["ID", self.input_column, self.target_column, self.reliability_column]
        ]
        print(column_df.head())
        merged_df = pd.merge(df, column_df, on=["ID"], how="left", suffixes=("", "_y"))
        merged_df = merged_df[
            merged_df.columns.drop(list(merged_df.filter(regex="_y")))
        ]
        output_dir = f"./data/outputs"
        os.makedirs(output_dir, exist_ok=True)
        merged_df.to_csv(
            f"{output_dir}/{self.input_column}_to_{self.target_column}_epoch_{self.epoch}_{self.timestamp}.csv",
            index=False,
        )
        return column_df
