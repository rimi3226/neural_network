#ifndef LAYER_H
#define LAYER_H

#include "Node.h"

// 레이어 구조체 정의
typedef struct Layer {
    int layer_size;           // 레이어 크기
    Node* nodes;              // 노드 배열
    double* dropout_mask;     // 드롭아웃 마스크
    double learning_rate;     // 학습률
    double dropout_rate;      // 드롭아웃 비율
} Layer;

// 레이어 초기화
void init_layer(Layer* layer, int layer_size, int next_layer_size, double learning_rate, double dropout_rate);

// 레이어 가중치 초기화
void set_layer_weights(Layer* layer, const char* weight_init);

// 입력 데이터 설정
void set_layer_values(Layer* layer, double* values);

// 순전파 계산
double* forward_layer(Layer* layer, const char* activation);

// 출력층 그래디언트 계산 (MSE)
void calculate_output_gradient_mse(Layer* layer, double* target);

// 출력층 그래디언트 계산 (Cross Entropy)
void calculate_output_gradient_ce(Layer* layer, double* target);

// 레이어 그래디언트 계산
void calculate_layer_gradient(Layer* layer, Layer* next_layer);

// 가중치 갱신
void update_layer_weights(Layer* layer, Layer* prev_layer);

// 레이어 메모리 해제
void free_layer(Layer* layer);

// 활성화 함수들
double sigmoid(double x);
double sigmoid_grad(double x);
double relu(double x);
double relu_grad(double x);

#endif
