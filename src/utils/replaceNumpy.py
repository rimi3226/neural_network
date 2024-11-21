import random
import math

def zeros(size):
    """0으로 가득찬 행렬 생성

    Args:
        size (int): 행렬 크기

    Returns:
        list: 0.0이 size만큼 있는 행렬
    """
    return [0.0] * size

def random_matrix(rows, cols):
    """랜덤값 행렬 생성

    Args:
        rows (int): 행 크기
        cols (int): 열 크기

    Returns:
        matrix : 랜덤값 채워진 행렬
    """
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

def multiply_matrix(matrix_a, matrix_b):
    """행렬곱

    Args:
        matrix_a (matrix): 행렬1
        matrix_b (matrix): 행렬2

    Raises:
        ValueError: 행렬 크기 서로 안 맞을 때

    Returns:
        matrix: 행렬곱 결과
    """
    # # 입력 데이터를 float으로 변환
    matrix_a = [[float(a) for a in row] for row in matrix_a]
    matrix_b = [[float(b) for b in row] for row in matrix_b]

    # 크기 검증
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("행렬곱 오류: 크기 불일치")

    # 행렬 곱 계산
    result = [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*matrix_b)]
        for row in matrix_a
    ]

    return result


def transpose(matrix):
    """행렬전치

    Args:
        matrix (matrix): 전치시킬 행렬

    Returns:
        matrix: 전치된 행렬
    """
    return [list(row) for row in zip(*matrix)]

def maximum(a, b):
    """최댓값 계산

    Args:
        a (list or float): 첫 번째 값 또는 리스트
        b (list or float): 두 번째 값 또는 리스트

    Returns:
        list or float: 두 값의 최대값
    """
    if isinstance(a, list) and isinstance(b, list):
        return [max(x, y) for x, y in zip(a, b)]
    elif isinstance(a, list):
        return [max(x, b) for x in a]
    elif isinstance(b, list):
        return [max(a, y) for y in b]
    else:
        return max(a, b)

def exp(matrix):
    """지수 함수 적용

    Args:
        matrix (list): 입력 행렬 (1D 또는 2D)

    Returns:
        list: 각 요소에 e^x를 적용한 결과
    """
    if isinstance(matrix[0], (float, int)):  # 1차원 리스트
        return [math.exp(x) for x in matrix]
    return [[math.exp(x) for x in row] for row in matrix]


def tanh(matrix):
    """하이퍼볼릭 탄젠트 함수 적용

    Args:
        matrix (list): 입력 행렬

    Returns:
        list: 각 요소에 tanh(x)를 적용한 결과
    """
    return [[math.tanh(x) for x in row] for row in matrix]