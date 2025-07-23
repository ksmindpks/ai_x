# simple_image_predictor.py
# 근본 기능만 하는 간단한 음식 이미지 예측기

import os
import glob
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PIL import Image
from io import BytesIO
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res_pre
import warnings
warnings.filterwarnings('ignore')

# 기본 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

IMG_SIZE = (224, 224)

class SimpleFoodPredictor:
    def __init__(self, model_dir='models', timestamp='20250716_144252'):
        self.model_dir = model_dir
        self.timestamp = timestamp
        self.load_models()
    
    def build_model(self, base_cls, num_classes):
        """모델 구조 빌드"""
        base = base_cls(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        x = GlobalAveragePooling2D(name='gap')(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=base.input, outputs=out)
    
    def load_models(self):
        """모델 로드"""
        # 라벨 매핑
        label_map = joblib.load(f"{self.model_dir}/label_to_index_{self.timestamp}.joblib")
        self.index_to_label = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)
        
        # EfficientNet
        self.eff_model = self.build_model(EfficientNetB0, num_classes)
        self.eff_model.load_weights(f"{self.model_dir}/effnet_model_best_{self.timestamp}.h5")
        
        # ResNet
        self.res_model = self.build_model(ResNet50V2, num_classes)
        self.res_model.load_weights(f"{self.model_dir}/resnet_model_best_{self.timestamp}.h5")
        
        # XGBoost
        try:
            self.xgb_model = joblib.load(f"{self.model_dir}/xgb_model_{self.timestamp}.joblib")
        except:
            self.xgb_model = None
        
        print(f"모델 로드 완료 - 클래스 수: {num_classes}")
    
    def download_image(self, url):
        """URL에서 이미지 다운로드"""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except:
            return None
    
    def predict_image(self, pil_image):
        """이미지 예측"""
        try:
            # 전처리
            img_array = np.array(pil_image)
            img = tf.image.resize(img_array, IMG_SIZE)
            img = tf.expand_dims(img, axis=0)
            
            # CNN 예측
            inp_eff = eff_pre(img)
            inp_res = res_pre(img)
            p_eff = self.eff_model.predict(inp_eff, verbose=0)
            p_res = self.res_model.predict(inp_res, verbose=0)
            p_cnn = (p_eff + p_res) / 2.0
            
            # XGBoost 예측 (있는 경우)
            if self.xgb_model:
                try:
                    feat_eff = Model(self.eff_model.input, self.eff_model.get_layer('gap').output).predict(inp_eff, verbose=0)
                    feat_res = Model(self.res_model.input, self.res_model.get_layer('gap').output).predict(inp_res, verbose=0)
                    feat = np.hstack([feat_eff, feat_res])
                    p_xgb = self.xgb_model.predict_proba(feat)
                    ensemble = p_cnn * 0.6 + p_xgb * 0.4
                except:
                    ensemble = p_cnn
            else:
                ensemble = p_cnn
            
            # 결과
            ensemble = ensemble.flatten()
            idx = np.argmax(ensemble)
            predicted_label = self.index_to_label[idx]
            confidence = float(ensemble[idx])
            
            return predicted_label, confidence
        except:
            return None, 0.0

def process_excel_files(folder_path):
    """Excel 파일들 처리"""
    # 예측기 초기화
    predictor = SimpleFoodPredictor()
    
    # Excel 파일 찾기
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    print(f"Excel 파일 {len(excel_files)}개 발견")
    
    for excel_file in excel_files:
        print(f"\n처리 중: {os.path.basename(excel_file)}")
        
        # Excel 읽기
        xl_file = pd.ExcelFile(excel_file)
        
        for sheet_name in xl_file.sheet_names:
            if '요약' in sheet_name:
                continue
                
            print(f"  시트: {sheet_name}")
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # URL 컬럼 찾기
            url_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['url', 'image', '이미지']):
                    url_col = col
                    break
            
            if not url_col:
                print(f"    URL 컬럼 없음")
                continue
            
            print(f"    URL 컬럼: {url_col}")
            
            # 예측 결과 컬럼 추가
            df['예측_음식명'] = ''
            df['예측_신뢰도'] = 0.0
            
            # 각 URL 처리
            count = 0
            success = 0
            
            for idx, row in df.iterrows():
                url = row[url_col]
                if pd.isna(url) or str(url).strip() == '':
                    continue
                
                count += 1
                
                # 이미지 다운로드
                pil_image = predictor.download_image(url)
                if pil_image is None:
                    continue
                
                # 예측
                predicted_food, confidence = predictor.predict_image(pil_image)
                if predicted_food:
                    df.loc[idx, '예측_음식명'] = predicted_food
                    df.loc[idx, '예측_신뢰도'] = round(confidence, 3)
                    success += 1
                    print(f"    [{success}/{count}] {predicted_food} ({confidence:.3f})")
            
            # 결과 저장
            with pd.ExcelWriter(excel_file, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"    완료: {success}/{count}개 성공")

def main():
    """메인 함수"""
    folder_path = r'C:\ai_x\source\Pks_Develop\N시기별음식URL수집\N월별조회메뉴수집_비중반영'
    
    if not os.path.exists(folder_path):
        print(f"폴더 없음: {folder_path}")
        return
    
    if not os.path.exists('models'):
        print("models 폴더 없음")
        return
    
    process_excel_files(folder_path)
    print("\n처리 완료!")

if __name__ == '__main__':
    main()