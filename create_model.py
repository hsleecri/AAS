import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np

# 모델 정의
X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, 1])
Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, 1])

W = onnx.helper.make_tensor('W', onnx.TensorProto.FLOAT, [1], np.array([2.0]).astype(np.float32))
B = onnx.helper.make_tensor('B', onnx.TensorProto.FLOAT, [1], np.array([1.0]).astype(np.float32))

node = onnx.helper.make_node(
    'Gemm',
    inputs=['X', 'W', 'B'],
    outputs=['Y'],
    alpha=1.0,
    beta=1.0,
    transA=0,
    transB=0
)

graph = onnx.helper.make_graph(
    [node],
    'simple_graph',
    [X],
    [Y],
    initializer=[W, B]
)

model = onnx.helper.make_model(graph)
onnx.save(model, 'simple_model.onnx')
