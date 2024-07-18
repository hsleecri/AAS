import onnx
import onnxruntime as ort
import numpy as np

# ONNX 모델 파일 로드
model_path = "mlp_classification.onnx"
model = onnx.load(model_path)

# 모델 검증
onnx.checker.check_model(model)
print("The model is valid!")

# ONNX Runtime을 사용하여 모델 실행
ort_session = ort.InferenceSession(model_path)

# 입력 데이터 준비 (예시 입력 데이터)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# 더 많은 입력 데이터 준비 (10개의 샘플, 각 샘플은 3개의 피처를 가짐)
input_data = np.random.rand(10, 3).astype(np.float32)  # 입력 크기: [10, 3]

# 모델 추론 실행
outputs = ort_session.run([output_name], {input_name: input_data})

# 예측 결과 출력
predictions = outputs[0]
print("Model inference output:", predictions)

# 이진 분류 결과 출력
predicted_classes = (predictions > 0.5).astype(np.int32)  # np.int 대신 np.int32 사용
print("Predicted classes:", predicted_classes)
