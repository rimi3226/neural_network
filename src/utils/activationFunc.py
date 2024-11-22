import replaceNumpy as rnp
import math
import numpy as np

def sigmoid(matrix):
    """
    시그모이드 활성화 함수 적용

    Args:
        matrix (list): 입력 행렬

    Returns:
        list: 각 요소에 sigmoid(x)를 적용한 결과
    """
    # sigmoid(x) = 1 / (1 + exp(-x))
    exp_neg_matrix = rnp.exp([[-x for x in row] for row in matrix])  # exp(-x) 계산
    return [[1 / (1 + exp_neg) for exp_neg in row] for row in exp_neg_matrix]

def tanh(matrix):
    """
    하이퍼볼릭 탄젠트 활성화 함수 적용

    Args:
        matrix (list): 입력 행렬

    Returns:
        list: 각 요소에 tanh(x)를 적용한 결과
    """
    return rnp.tanh(matrix)  # replaceNumpy의 tanh 함수 활용

def relu(matrix):
    """
    ReLU 활성화 함수 적용

    Args:
        matrix (list): 입력 행렬

    Returns:
        list: 각 요소에 ReLU(x)를 적용한 결과
    """
    return [[rnp.maximum(x, 0) for x in row] for row in matrix]  # max(x, 0)

def leaky_relu(matrix, alpha=0.01):
    """
    Leaky ReLU 활성화 함수 적용

    Args:
        matrix (list): 입력 행렬
        alpha (float): 기울기 (default=0.01)

    Returns:
        list: 각 요소에 Leaky ReLU(x)를 적용한 결과
    """
    return [[x if x > 0 else alpha * x for x in row] for row in matrix]  # Leaky ReLU 구현

# def softmax(matrix):
#     """
#     소프트맥스 활성화 함수 적용

#     Args:
#         matrix (list): 입력 벡터(1D) 또는 행렬(2D)

#     Returns:
#         list: 각 요소에 softmax(x)를 적용한 결과
#     """
#     # 1차원 리스트 처리
#     if isinstance(matrix[0], (float, int)):  # 1D case
#         max_value = max(matrix)  # 최대값
#         exp_row = [math.exp(x - max_value) for x in matrix]  # exp(x - max)
#         sum_exp_row = sum(exp_row)  # exp 합계
#         return [value / sum_exp_row for value in exp_row]  # 확률값 계산

#     # 2차원 리스트 처리
#     softmax_result = []
#     for row in matrix:
#         max_value = max(row)  # 최대값
#         exp_row = [math.exp(x - max_value) for x in row]  # exp(x - max)
#         sum_exp_row = sum(exp_row)  # exp 합계
#         softmax_result.append([value / sum_exp_row for value in exp_row])  # 확률값 계산
#     return softmax_result
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()