#include <stdio.h>
#include <stdlib.h>
#include "CSVParser.h"
#include "NeuralNetwork.h"

// gcc -o neural_net main.c CSVParser.c NeuralNetwork.c Layer.c Node.c -lm

int main() {
    const char* filename = "network.csv"; // CSV 파일 이름
    int layer_count;

    // CSV 파일에서 레이어 데이터를 읽음
    Layer* layers = parse_csv(filename, &layer_count);

    if (!layers) {
        fprintf(stderr, "Error: Failed to parse layers from CSV.\n");
        return EXIT_FAILURE;
    }

    printf("Successfully parsed %d layers from CSV.\n", layer_count);
    print_layers(layers, layer_count);

    // NeuralNetwork 초기화
    NeuralNetwork nn;
    init_neural_network(
        &nn,                                // 신경망 구조체
        layers[0].layer_size,               // 입력층 크기 (첫 레이어 크기)
        (int[]){layers[1].layer_size},      // 은닉층 크기 배열 (예: 두 번째 레이어 크기)
        layer_count - 2,                    // 은닉층 개수
        layers[layer_count - 1].layer_size, // 출력층 크기 (마지막 레이어 크기)
        0.01,                               // 학습률
        "relu",                             // 은닉층 활성화 함수
        "softmax",                          // 출력층 활성화 함수
        "He",                               // 가중치 초기화 방법
        "adam",                             // 최적화 방법
        0.5,                                // 드롭아웃 비율
        "mse"                               // 손실 함수
    );

    // CSV에서 읽은 가중치를 신경망에 설정
    for (int i = 0; i < nn.num_layers; i++) {
        nn.layers[i] = layers[i];
    }

    // 입력 데이터 설정 (예제 데이터)
    // double input_data[] = {1.0, 2.0, 3.0}; // 입력층 데이터
    // set_input_layer(&nn, input_data);

    // 순전파 수행
    forward_propagation(&nn);

    // 출력 결과 확인
    printf("Forward propagation result:\n");
    print_layers(nn.layers, nn.num_layers);

    // 메모리 해제
    free_neural_network(&nn);
    free_layers(layers, layer_count);

    return 0;
}
