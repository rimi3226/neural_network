import pandas as pd
import numpy as np

def preprocess_data(csv_path):
    # 데이터 불러오기
    data = pd.read_csv(csv_path, header=None).values  # CSV 파일 읽기

    # 결측값 처리: '-999'를 float으로 변환
    processed_data = []
    for row in data:
        processed_row = [float(x) if x != '-999' else -999 for x in row]
        processed_data.append(processed_row)
    processed_data = np.array(processed_data)

    # 타겟값과 입력값 분리
    targets = processed_data[:, 0].astype(int)  # 0번째 열: 타겟값
    inputs = processed_data[:, 1:]  # 1번째부터 끝까지: 입력값

    # 입력 데이터 정규화 (Min-Max Scaling)
    inputs = (inputs - inputs.min(axis=1, keepdims=True)) / (inputs.max(axis=1, keepdims=True) - inputs.min(axis=1, keepdims=True) + 1e-8)

    # 원-핫 인코딩 타겟값 생성
    num_classes = len(np.unique(targets))
    targets_one_hot = np.eye(num_classes)[targets]

    return inputs, targets_one_hot, targets
