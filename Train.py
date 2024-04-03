# 모델 최종본
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split


class CustomBertForSequenceClassification(BertForSequenceClassification):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
            output_hidden_states=True
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels,
            output_hidden_states=output_hidden_states
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-5]  # n번째 레이어의 hidden states를 반환합니다.
        loss = outputs.loss
        return logits, loss, hidden_states


# 데이터 로드 및 전처리
data_A = pd.read_csv("Train_data.csv")  # data set A 파일명에 맞게 수정
data_B = pd.read_csv("infected.csv")  # data set B 파일명에 맞게 수정
# 모델 저장 경로
model_path = "Pre-trained.pt"

# 중복된 환자 정보 제거
data_A_unique = data_A.drop_duplicates(subset="ID")

# X_train, Y_train 생성
X_train = []
Y_train = []

for index, row in data_A_unique.iterrows():
    patient_id = row["ID"]
    patient_info = [str(row[column]) for column in data_A.columns if column != "ID" and column != "DESCRIPTION"]
    symptoms = ", ".join(data_A[data_A["ID"] == patient_id]["DESCRIPTION"].tolist())
    combined_info = ", ".join(patient_info) + ", " + symptoms
    X_train.append(combined_info)
    if patient_id in data_B.values:
        Y_train.append(1)
    else:
        Y_train.append(0)

# 라벨확인
print("X_train\n", X_train[:10])
print("Y_train\n", Y_train[:10])

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 모델이 이미 저장되어 있는지 확인하고, 저장된 모델이 있으면 불러오고 없으면 새로운 모델 생성
if os.path.exists(model_path):
    # 저장된 모델이 있을 경우 불러오기
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path))
    print("Pre-train model loaded.")
else:
    # 저장된 모델이 없을 경우 새로운 모델 생성
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    print("New model generated.")

# 입력 데이터를 BERT의 입력 형식으로 변환
max_len = 128  # 입력 시퀀스의 최대 길이

input_ids = []
attention_masks = []

for info in X_train:
    encoded_dict = tokenizer.encode_plus(
        info,  # 환자 정보 및 증상
        add_special_tokens=True,  # [CLS], [SEP] 토큰 추가
        max_length=max_len,  # 최대 길이 지정
        pad_to_max_length=True,  # 패딩을 추가하여 최대 길이로 맞춤
        return_attention_mask=True,  # 어텐션 마스크 생성
        return_tensors='pt',  # PyTorch 텐서로 반환
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(Y_train)

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = 0.8
train_dataset, val_dataset = train_test_split(dataset, test_size=1 - train_size, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 모델을 GPU로 이동
model.to(device)

# 옵티마이저 및 학습률 설정
# 기본 학습률 : 2e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)

# 에폭 설정
epochs = 3

# 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    hidden_states_list = []  # 각 배치의 hidden state를 저장할 리스트
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[1]  # loss가 outputs의 두 번째 값입니다.
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Batch Loss: {loss.item()}')
        # hidden state를 저장합니다.
        hidden_states = outputs[2]
        hidden_states_list.append(hidden_states)

    # 각 배치의 hidden state를 합쳐서 CSV 파일로 저장합니다.
    hidden_states = torch.cat(hidden_states_list, dim=0)
    hidden_states = hidden_states[:, 0, :].cpu().detach().numpy()
    hidden_states_df = pd.DataFrame(hidden_states)
    hidden_states_df.to_csv(f"hidden_states_epoch{epoch + 1}.csv", index=False)  # 각 epoch마다 파일 이름을 다르게 지정합니다.

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')

# 모델 저장
torch.save(model.state_dict(), model_path)

# 모델 평가
model.eval()
val_accuracy = 0
for batch in val_dataloader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs[0]  # logits가 outputs의 첫 번째 값입니다.
    logits = logits.detach().cpu().numpy()
    label_ids = inputs['labels'].cpu().numpy()
    val_accuracy += (logits.argmax(axis=1) == label_ids).mean().item()

print(f'Validation Accuracy: {val_accuracy / len(val_dataloader)}')
