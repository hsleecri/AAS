import onnx
import onnxruntime as ort
import numpy as np

# ONNX 모델 파일 로드
model_path = "mlp_model.onnx"
model = onnx.load(model_path)

# 모델 검증
onnx.checker.check_model(model)
print("The model is valid!")

# ONNX Runtime을 사용하여 모델 실행
ort_session = ort.InferenceSession(model_path)

# 입력 데이터 준비 (예시 입력 데이터)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
input_data = np.random.rand(1, 3).astype(np.float32)  # 입력 크기: [1, 3]

# 모델 추론 실행
outputs = ort_session.run([output_name], {input_name: input_data})

print("Model inference output:", outputs)
