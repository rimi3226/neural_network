#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CSVParser.h"
#include "Node.h"
#include "Layer.h"

// CSV 파싱 함수
Layer* parse_csv(const char* filename, int* layer_count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    char line[4096];
    Layer* layers = NULL;
    *layer_count = 0;

    // 헤더 스킵
    if (fgets(line, sizeof(line), file) == NULL) {
        perror("Error reading header");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    while (fgets(line, sizeof(line), file)) {
        int layer_index, node_index;
        char* weights_str;

        // CSV 데이터 분리
        char* token = strtok(line, ",");
        if (!token) continue;
        layer_index = atoi(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        node_index = atoi(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        weights_str = token;

        // 레이어 배열 확장
        if (layer_index >= *layer_count) {
            layers = realloc(layers, sizeof(Layer) * (layer_index + 1));
            if (!layers) {
                fprintf(stderr, "Memory allocation failed for layers.\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            for (int i = *layer_count; i <= layer_index; i++) {
                init_layer(&layers[i], 0, 0, 0.0, 0.0); // 초기 빈 레이어 생성
            }
            *layer_count = layer_index + 1;
        }

        // 레이어 가져오기
        Layer* layer = &layers[layer_index];

        // 노드 배열 확장
        if (node_index >= layer->layer_size) {
            int new_node_count = node_index + 1;
            layer->nodes = realloc(layer->nodes, sizeof(Node) * new_node_count);
            if (!layer->nodes) {
                fprintf(stderr, "Memory allocation failed for nodes.\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            for (int i = layer->layer_size; i < new_node_count; i++) {
                init_node(&layer->nodes[i], 0, "relu"); // 초기 빈 노드 생성
            }
            layer->layer_size = new_node_count;
        }

        // 가중치 파싱
        Node* node = &layer->nodes[node_index];
        char* weight_token = strtok(weights_str, " []");
        while (weight_token) {
            node->weight = realloc(node->weight, sizeof(double) * (node->next_layer_size + 1));
            if (!node->weight) {
                fprintf(stderr, "Memory allocation failed for weights.\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            node->weight[node->next_layer_size++] = strtod(weight_token, NULL);
            weight_token = strtok(NULL, " []");
        }
    }

    fclose(file);
    return layers;
}

// 레이어 메모리 해제 함수
void free_layers(Layer* layers, int layer_count) {
    for (int i = 0; i < layer_count; i++) {
        free_layer(&layers[i]);
    }
    free(layers);
}

// 레이어 정보 출력 함수
void print_layers(const Layer* layers, int layer_count) {
    for (int i = 0; i < layer_count; i++) {
        printf("Layer %d:\n", i);
        for (int j = 0; j < layers[i].layer_size; j++) {
            printf("  Node %d: Weights [", j);
            for (int k = 0; k < layers[i].nodes[j].next_layer_size; k++) {
                printf("%.6f", layers[i].nodes[j].weight[k]);
                if (k < layers[i].nodes[j].next_layer_size - 1) {
                    printf(", ");
                }
            }
            printf("]\n");
        }
    }
}
