import matplotlib.pyplot as plt

# 네트워크의 각 레이어 노드와 가중치를 출력
def print_network(layers):
    for i, layer in enumerate(layers):
        print(f"Layer {i + 1}:")
        print(f"  Nodes: {layer.nodes}")
        if layer.weights:
            print(f"  Weights: {layer.weights}")

# 네트워크의 각 레이어 노드 값을 출력
def print_layer_nodes(layers):
    print("=" * 50)
    print("Neural Network Layer Nodes:")
    for i, layer in enumerate(layers):
        print(f"Layer {i + 1}:")
        print(f"  Nodes: {layer.nodes}")
    print("=" * 50)

# 각 은닉층의 가중치를 히스토그램으로 시각화
def plot_weights(layers):
    for i, layer in enumerate(layers[1:], start=1):  # 입력층 제외
        if layer.weights:
            flattened_weights = [w for row in layer.weights for w in row]  # 2차원 가중치를 1차원으로 변환
            plt.figure(figsize=(8, 4))
            plt.hist(flattened_weights, bins=50, alpha=0.75, color='blue', edgecolor='black')
            plt.title(f"Layer {i}: Weight Distribution")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

# 특정 Layer의 노드 값을 선 그래프로 시각화
def plot_layer_nodes(nodes):
    # Flatten if 2D array
    if isinstance(nodes[0], list):  # 2D array
        flat_nodes = [node for row in nodes for node in row]
    else:  # 1D array
        flat_nodes = nodes

    # 선 그래프 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(flat_nodes, marker='o', linestyle='-', alpha=0.75)
    plt.title("Node Value Distribution")
    plt.xlabel("Node Index")
    plt.ylabel("Node Value")
    plt.grid(True)
    plt.show()
