#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "NeuralNetwork.h"

// NeuralNetwork 초기화
void init_neural_network(NeuralNetwork* nn, int input_size, int* hidden_sizes, int hidden_count,
                         int output_size, double learning_rate, const char* activation,
                         const char* output_activation, const char* weight_init,
                         const char* optimizer, double dropout_rate, const char* loss_function) {
    nn->learning_rate = learning_rate;
    nn->num_layers = hidden_count + 2; // 입력층 + 은닉층 + 출력층
    nn->layers = (Layer*)malloc(sizeof(Layer) * nn->num_layers);
    strncpy(nn->activation, activation, sizeof(nn->activation) - 1);
    strncpy(nn->output_activation, output_activation, sizeof(nn->output_activation) - 1);
    strncpy(nn->loss_function, loss_function, sizeof(nn->loss_function) - 1);
    nn->dropout_rate = dropout_rate;

    // 입력층 초기화
    init_layer(&nn->layers[0], input_size, hidden_sizes[0], learning_rate, dropout_rate);
    set_layer_weights(&nn->layers[0], weight_init);

    // 은닉층 초기화
    for (int i = 0; i < hidden_count; i++) {
        int next_layer_size = (i + 1 < hidden_count) ? hidden_sizes[i + 1] : output_size;
        init_layer(&nn->layers[i + 1], hidden_sizes[i], next_layer_size, learning_rate, dropout_rate);
        set_layer_weights(&nn->layers[i + 1], weight_init);
    }

    // 출력층 초기화
    init_layer(&nn->layers[nn->num_layers - 1], output_size, 0, learning_rate, dropout_rate);
}

// 입력층 설정
void set_input_layer(NeuralNetwork* nn, double* data) {
    set_layer_values(&nn->layers[0], data);
}

// 타겟 레이어 설정
void set_target_layer(NeuralNetwork* nn, int target_index) {
    nn->target_layer = (double*)calloc(nn->layers[nn->num_layers - 1].layer_size, sizeof(double));
    if (target_index < nn->layers[nn->num_layers - 1].layer_size) {
        nn->target_layer[target_index] = 1.0;
    }
}

// 순전파
void forward_propagation(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        double* activation_values = forward_layer(&nn->layers[i], nn->activation);
        set_layer_values(&nn->layers[i + 1], activation_values);
        free(activation_values);
    }

    // 출력층 활성화 함수 적용
    double* activation_values = forward_layer(&nn->layers[nn->num_layers - 1], nn->output_activation);
    set_layer_values(&nn->layers[nn->num_layers - 1], activation_values);
    free(activation_values);
}

// 역전파
void backward_propagation(NeuralNetwork* nn) {
    Layer* output_layer = &nn->layers[nn->num_layers - 1];

    if (strcmp(nn->loss_function, "mse") == 0) {
        calculate_output_gradient_mse(output_layer, nn->target_layer);
    } else if (strcmp(nn->loss_function, "cross_entropy") == 0) {
        calculate_output_gradient_ce(output_layer, nn->target_layer);
    }

    for (int i = nn->num_layers - 2; i > 0; i--) {
        calculate_layer_gradient(&nn->layers[i], &nn->layers[i + 1]);
        update_layer_weights(&nn->layers[i], &nn->layers[i - 1]);
    }
}

// 메모리 해제
void free_neural_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        free_layer(&nn->layers[i]);
    }
    free(nn->layers);
    free(nn->target_layer);
}
