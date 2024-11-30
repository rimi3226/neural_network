#ifndef NODE_H
#define NODE_H

// 노드 구조체 정의
typedef struct Node {
    double val;               // 노드 값
    double* weight;           // 가중치 배열
    double bias;              // 편향
    double gradient;          // 그래디언트
    double activation_gradient; // 활성화 함수 미분값
    int next_layer_size;      // 다음 레이어 크기
    char activation[10];      // 활성화 함수 이름
} Node;

// 노드 초기화
void init_node(Node* node, int next_layer_size, const char* activation);

// 가중치 초기화
void init_weight(Node* node, int next_layer_size, const char* weight_init);

// 노드 메모리 해제
void free_node(Node* node);

#endif
