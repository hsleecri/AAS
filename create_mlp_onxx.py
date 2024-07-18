import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# MLP 모델 정의
# 입력과 출력 정의
input = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [None, 3])
output = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [None, 2])

# 가중치와 바이어스 초기화
W1 = numpy_helper.from_array(np.random.rand(3, 4).astype(np.float32), name='W1')
B1 = numpy_helper.from_array(np.random.rand(4).astype(np.float32), name='B1')
W2 = numpy_helper.from_array(np.random.rand(4, 2).astype(np.float32), name='W2')
B2 = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='B2')

# 레이어 정의
node1 = helper.make_node(
    'Gemm',
    inputs=['input', 'W1', 'B1'],
    outputs=['hidden'],
)
node2 = helper.make_node(
    'Relu',
    inputs=['hidden'],
    outputs=['relu_hidden'],
)
node3 = helper.make_node(
    'Gemm',
    inputs=['relu_hidden', 'W2', 'B2'],
    outputs=['output'],
)

# 그래프 생성
graph_def = helper.make_graph(
    [node1, node2, node3],
    'mlp',
    [input],
    [output],
    initializer=[W1, B1, W2, B2],
)

# 모델 생성, opset 버전을 13으로 설정
model_def = helper.make_model(graph_def, producer_name='mlp-example', opset_imports=[helper.make_opsetid("", 13)])

# 모델 저장
onnx.save(model_def, 'mlp_model.onnx')
print("MLP 모델이 'mlp_model.onnx'에 저장되었습니다.")