#ifndef CSVPARSER_H
#define CSVPARSER_H

#include "Layer.h"

// CSV 파싱 함수: CSV 파일을 읽어 Layer 구조체 배열을 반환
Layer* parse_csv(const char* filename, int* layer_count);

// 레이어 메모리 해제 함수
void free_layers(Layer* layers, int layer_count);

// 레이어 정보 출력 함수
void print_layers(const Layer* layers, int layer_count);

#endif // CSVPARSER_H
