#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Node.h"

// 노드 초기화
void init_node(Node* node, int next_layer_size, const char* activation) {
    node->val = 0.0;
    node->gradient = 0.0;
    node->activation_gradient = 1.0; // 기본값
    node->next_layer_size = next_layer_size;
    node->bias = 0.0;
    node->weight = (double*)malloc(sizeof(double) * next_layer_size);
    if (node->weight == NULL) {
        fprintf(stderr, "Memory allocation failed for weights.\n");
        exit(EXIT_FAILURE);
    }
    strncpy(node->activation, activation, sizeof(node->activation) - 1);
    node->activation[sizeof(node->activation) - 1] = '\0';
}

// 가중치 초기화
void init_weight(Node* node, int next_layer_size, const char* weight_init) {
    double scale = 1.0;

    // 초기화 방식에 따른 스케일 계산
    if (strcmp(weight_init, "He") == 0) {
        scale = sqrt(2.0 / next_layer_size);
    } else if (strcmp(weight_init, "Xavier") == 0) {
        scale = sqrt(1.0 / next_layer_size);
    } else {
        scale = atof(weight_init); // 사용자 정의 값
    }

    // 가중치 및 편향 초기화
    for (int i = 0; i < next_layer_size; i++) {
        node->weight[i] = scale * ((double)rand() / RAND_MAX * 2.0 - 1.0); // [-1, 1] 범위의 랜덤 값
    }
    node->bias = scale * ((double)rand() / RAND_MAX * 2.0 - 1.0); // [-1, 1] 범위의 랜덤 편향 값
}

// 노드 메모리 해제
void free_node(Node* node) {
    if (node->weight != NULL) {
        free(node->weight);
        node->weight = NULL;
    }
}
