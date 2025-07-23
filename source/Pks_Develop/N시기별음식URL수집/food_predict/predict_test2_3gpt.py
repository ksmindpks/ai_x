# predict.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # '2' 이상(INFO/WARNING) 메시지 억제
import glob
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Keras retracing 경고도 억제
import joblib
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet_v2   import preprocess_input as res_pre

IMG_SIZE = (224, 224)

def build_transfer_model(base_cls, num_classes, dropout_rate=0.2):
    base = base_cls(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D(name='gap')(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return Model(inputs=base.input, outputs=out)

def load_models(model_dir: str, timestamp: str):
    label_map = joblib.load(os.path.join(model_dir, f"label_to_index_{timestamp}.joblib"))
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    eff = build_transfer_model(EfficientNetB0, num_classes, dropout_rate=0.2)
    eff.load_weights(os.path.join(model_dir, f"effnet_model_best_{timestamp}.h5"))

    res = build_transfer_model(ResNet50V2, num_classes, dropout_rate=0.2)
    res.load_weights(os.path.join(model_dir, f"resnet_model_best_{timestamp}.h5"))

    xgb = joblib.load(os.path.join(model_dir, f"xgb_model_{timestamp}.joblib"))

    return eff, res, xgb, index_to_label

def predict_image(path: str,
                  eff_model,
                  res_model,
                  xgb_model,
                  index_to_label: dict):
    raw = tf.io.read_file(path)
    # 주요 비트맵 포맷 자동 디코딩
    try:
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError:
        img = tf.image.decode_webp(raw, channels=3)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    inp_eff = eff_pre(img)[None, ...]
    inp_res = res_pre(img)[None, ...]

    p_eff = eff_model.predict(inp_eff, verbose=0)
    p_res = res_model.predict(inp_res, verbose=0)
    p_cnn = (p_eff + p_res) / 2.0

    feat_eff = Model(eff_model.input, eff_model.get_layer('gap').output)\
                .predict(inp_eff, verbose=0)
    feat_res = Model(res_model.input, res_model.get_layer('gap').output)\
                .predict(inp_res, verbose=0)
    feat = np.hstack([feat_eff, feat_res])
    p_xgb = xgb_model.predict_proba(feat)

    # 앙상블 가중 평균
    ensemble = p_cnn * 0.6 + p_xgb * 0.4
    ensemble = ensemble.flatten()  # shape (num_classes,)

    # 가장 높은 확률과 라벨
    idx = int(np.argmax(ensemble))
    label = index_to_label[idx]
    confidence = float(ensemble[idx])

    # 전체 확률을 {label:prob} dict로
    probs = { index_to_label[i]: float(ensemble[i]) for i in range(len(ensemble)) }

    return label, confidence, probs

def main():
    # ─── 변수 정의 ───────────────────────────────────────────
    MODEL_DIR = 'models' # ******** 모델 경로 ***********
    TIMESTAMP = '20250716_144252' # ******** 모델 명 ***********
    RAW_PATHS = [                 # ******** 예측파일 위치 ***********
        r'C:\ksmindpks\test_images'        # 폴더
        # r'C:\ksmindpks\deepDishData\banh_mi\png\banh_mi_0001.png' # 파일
    ]

    # ─── 이미지 경로 정리 ─────────────────────────────────────
    IMAGE_PATHS = []
    for p in RAW_PATHS:
        if os.path.isdir(p):
            for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.webp'):
                IMAGE_PATHS += glob.glob(os.path.join(p, ext))
        elif os.path.isfile(p):
            IMAGE_PATHS.append(p)
        else:
            print(f"경로 없음: {p}")

    # ─── 모델 로드 & 예측 루프 ─────────────────────────────────
    eff_model, res_model, xgb_model, idx2label = load_models(MODEL_DIR, TIMESTAMP)
    for img_path in IMAGE_PATHS:
        label, conf, probs = predict_image(img_path, eff_model, res_model, xgb_model, idx2label)
        print(f"\n=== {os.path.basename(img_path)} ===")
        print(f"Predicted: {label} (conf: {conf:.3f})")
        for lab, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {lab}: {p:.3f}")

if __name__ == '__main__':
    main() 
