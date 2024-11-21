import replaceNumpy as rnp
import math
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

def softmax(matrix):
    result = []
    for row in matrix:
        max_value = max(row)
        shifted_row = [x - max_value for x in row]  # 안정화
        exp_row = [math.exp(x) for x in shifted_row]
        sum_exp_row = sum(exp_row)
        result.append([value / sum_exp_row for value in exp_row])
    return result