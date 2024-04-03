import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# 추론 정확도 계산 (수정)
def calculate_accuracy(client_data, server_data, top_n_indices):
    correct_count = 0
    for i in range(len(client_data)):
        if i in top_n_indices[i]:
            correct_count += 1
    accuracy = correct_count / len(client_data)
    return accuracy

# 파일 경로
client_files = ['Client-trained_smashed_data_epoch1.csv', 'Client-trained_smashed_data_epoch2.csv', 'Client-trained_smashed_data_epoch3.csv']
server_files = ['Dictionary_smashed_data_epoch1.csv', 'Dictionary_smashed_data_epoch2.csv', 'Dictionary_smashed_data_epoch3.csv']

# 클라이언트 데이터 불러오기
client_data = np.concatenate([load_data(file) for file in client_files], axis=0)

# 서버 데이터 불러오기
server_data = np.concatenate([load_data(file) for file in server_files], axis=0)

# 정확도 계산을 위한 작업
total_correct_count = 0
total_clients = client_data.shape[0]

# 클라이언트 데이터마다 상위 5개의 추론 결과를 계산하고 정확도 측정
for i in range(total_clients):
    # 클라이언트 데이터 한 개에 대한 상위 5개의 추론 결과 계산
    client_single = client_data[i:i+1]
    top_n_indices_single = top_n_inference(client_single, server_data)
    # 클라이언트 데이터가 서버 데이터의 상위 5개 안에 포함되는지 확인
    if i in top_n_indices_single[0]:
        total_correct_count += 1

# 정확도 계산
accuracy = total_correct_count / total_clients
print("Top@5 Accuracy:", accuracy)
