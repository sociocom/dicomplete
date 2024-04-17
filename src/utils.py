import torch.nn as nn
from transformers import T5ForConditionalGeneration


class T5FineTuner(nn.Module):

    def __init__(self, pretrained_model_name: str = "sonoisa/t5-base-japanese"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )


def generate_text_from_model(
    tags,
    trained_model,
    tokenizer,
    num_return_sequences=1,
    max_length_src=30,
    max_length_target=300,
    num_beams=10,
    device="cpu",
):
    trained_model.eval()

    batch = tokenizer(
        tags,
        max_length=max_length_src,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # 生成処理を行う
    outputs = trained_model.model.generate(
        # outputs = trained_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length_target,
        repetition_penalty=8.0,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
        # temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
        num_beams=num_beams,  # ビームサーチの探索幅
        # diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        # num_beam_groups=10,  # ビームサーチのグループ
        num_return_sequences=num_return_sequences,  # 生成する文の数
    )

    generated_texts = [
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for ids in outputs
    ]

    return generated_texts


def write_to_csv(df, csv_filename):
    try:
        with open(csv_filename, "a") as f:
            df.to_csv(f, index=False, header=f.tell() == 0)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        pass
