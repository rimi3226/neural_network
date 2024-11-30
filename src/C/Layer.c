#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Layer.h"

// 활성화 함수들
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_grad(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double relu(double x) {
    return fmax(0, x);
}

double relu_grad(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// 레이어 초기화
void init_layer(Layer* layer, int layer_size, int next_layer_size, double learning_rate, double dropout_rate) {
    layer->layer_size = layer_size;
    layer->learning_rate = learning_rate;
    layer->dropout_rate = dropout_rate;
    layer->nodes = (Node*)malloc(sizeof(Node) * layer_size);
    layer->dropout_mask = (double*)malloc(sizeof(double) * layer_size);

    for (int i = 0; i < layer_size; i++) {
        init_node(&layer->nodes[i], next_layer_size, "relu");
        layer->dropout_mask[i] = 1.0;
    }
}

// 레이어 가중치 초기화
void set_layer_weights(Layer* layer, const char* weight_init) {
    for (int i = 0; i < layer->layer_size; i++) {
        init_weight(&layer->nodes[i], layer->nodes[i].next_layer_size, weight_init);
    }
}

// 입력 데이터 설정
void set_layer_values(Layer* layer, double* values) {
    for (int i = 0; i < layer->layer_size; i++) {
        layer->nodes[i].val = values[i];
    }
}

// 순전파 계산
double* forward_layer(Layer* layer, const char* activation) {
    double* output = (double*)malloc(sizeof(double) * layer->layer_size);
    for (int i = 0; i < layer->layer_size; i++) {
        Node* node = &layer->nodes[i];
        node->val = 0.0;
        for (int j = 0; j < node->next_layer_size; j++) {
            node->val += node->weight[j];
        }

        // 활성화 함수 적용
        if (strcmp(activation, "sigmoid") == 0) {
            node->val = sigmoid(node->val);
        } else if (strcmp(activation, "relu") == 0) {
            node->val = relu(node->val);
        }
        output[i] = node->val;
    }
    return output;
}

// 출력층 그래디언트 계산 (MSE)
void calculate_output_gradient_mse(Layer* layer, double* target) {
    for (int i = 0; i < layer->layer_size; i++) {
        Node* node = &layer->nodes[i];
        node->gradient = (node->val - target[i]) * node->activation_gradient;
    }
}

// 출력층 그래디언트 계산 (Cross Entropy)
void calculate_output_gradient_ce(Layer* layer, double* target) {
    for (int i = 0; i < layer->layer_size; i++) {
        Node* node = &layer->nodes[i];
        node->gradient = -target[i] / node->val;
    }
}

// 레이어 그래디언트 계산
void calculate_layer_gradient(Layer* layer, Layer* next_layer) {
    for (int i = 0; i < layer->layer_size; i++) {
        Node* node = &layer->nodes[i];
        node->gradient = 0.0;
        for (int j = 0; j < next_layer->layer_size; j++) {
            node->gradient += next_layer->nodes[j].weight[i] * next_layer->nodes[j].gradient;
        }
    }
}

// 가중치 갱신
void update_layer_weights(Layer* layer, Layer* prev_layer) {
    for (int i = 0; i < layer->layer_size; i++) {
        Node* node = &layer->nodes[i];
        for (int j = 0; j < node->next_layer_size; j++) {
            node->weight[j] -= layer->learning_rate * node->gradient * prev_layer->nodes[j].val;
        }
    }
}

// 레이어 메모리 해제
void free_layer(Layer* layer) {
    for (int i = 0; i < layer->layer_size; i++) {
        free_node(&layer->nodes[i]);
    }
    free(layer->nodes);
    free(layer->dropout_mask);
}
