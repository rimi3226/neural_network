#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Node.h"
#include "Layer.h"

// NeuralNetwork 구조체 정의
typedef struct NeuralNetwork {
    Layer* layers;            // 레이어 배열
    int num_layers;           // 레이어 개수
    double learning_rate;     // 학습률
    char activation[10];      // 활성화 함수
    char output_activation[10]; // 출력 활성화 함수
    double* target_layer;     // 목표 출력
    double dropout_rate;      // 드롭아웃 비율
    char loss_function[20];   // 손실 함수
} NeuralNetwork;

// NeuralNetwork 초기화
void init_neural_network(NeuralNetwork* nn, int input_size, int* hidden_sizes, int hidden_count,
                         int output_size, double learning_rate, const char* activation,
                         const char* output_activation, const char* weight_init,
                         const char* optimizer, double dropout_rate, const char* loss_function);

// 입력층 설정
void set_input_layer(NeuralNetwork* nn, double* data);

// 타겟 레이어 설정
void set_target_layer(NeuralNetwork* nn, int target_index);

// 순전파
void forward_propagation(NeuralNetwork* nn);

// 역전파
void backward_propagation(NeuralNetwork* nn);

// 메모리 해제
void free_neural_network(NeuralNetwork* nn);

#endif
