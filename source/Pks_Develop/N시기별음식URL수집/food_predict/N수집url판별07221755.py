# image_url_predictor.py

import os
import sys

# ========== 성능 최적화 환경 설정 ==========
# TensorFlow 로그 레벨 설정 (가장 먼저)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 모든 로그 억제
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'       # Intel oneDNN 최적화 활성화

# GPU 비활성화 (CPU 최적화에 집중)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# CPU 최적화 설정
os.environ['OMP_NUM_THREADS'] = '8'             # OpenMP 스레드 수 (CPU 코어 수에 맞게)
os.environ['TF_NUM_INTEROP_THREADS'] = '4'      # TensorFlow inter-op 스레드
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'      # TensorFlow intra-op 스레드

# 메모리 최적화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # GPU 메모리 증가 비활성화
os.environ['TF_MEMORY_ALLOCATION'] = 'BFC'      # Best-Fit with Coalescing 메모리 할당자

# Intel CPU 최적화 (Intel CPU인 경우)
os.environ['KMP_BLOCKTIME'] = '0'               # Intel OpenMP 대기 시간
os.environ['KMP_SETTINGS'] = '1'                # OpenMP 설정 출력
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

# Python 최적화
sys.dont_write_bytecode = True                  # .pyc 파일 생성 방지
# ============================================

import glob
import requests
import numpy as np
import pandas as pd

# NumPy 최적화
np.seterr(all='ignore')  # NumPy 경고 억제

import tensorflow as tf

# TensorFlow 최적화 설정 (import 직후)
tf.get_logger().setLevel('ERROR')

# Mixed Precision 활성화 (성능 향상)
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision 활성화 완료")
except:
    print("Mixed precision 비활성화 (구버전 TensorFlow)")

# TensorFlow 실행 최적화
tf.config.optimizer.set_jit(True)               # XLA JIT 컴파일 활성화

# CPU 스레드 설정
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(8)

# 메모리 최적화
tf.config.experimental.enable_memory_growth = False

import joblib
from PIL import Image
from io import BytesIO
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res_pre
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PIL 최적화
Image.MAX_IMAGE_PIXELS = None                   # PIL 이미지 크기 제한 해제
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# Requests 세션 최적화 (더 빠른 설정)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# 연결 재사용을 위한 어댑터 설정 (더 공격적인 설정)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=1,                    # 재시도 1회로 단축
    backoff_factor=0.05,        # 백오프 시간 단축
    status_forcelist=[500, 502, 503, 504],  # 429 제거 (속도 우선)
)
adapter = HTTPAdapter(
    pool_connections=20,        # 더 많은 연결
    pool_maxsize=50,           # 더 큰 풀 크기
    max_retries=retry_strategy
)
session.mount("http://", adapter)
session.mount("https://", adapter)

IMG_SIZE = (224, 224)

class FoodImagePredictor:
    """음식 이미지 URL 예측기 - 성능 최적화 버전"""
    
    def __init__(self, model_dir='models', timestamp='20250716_144252'):
        self.model_dir = model_dir
        self.timestamp = timestamp
        self.eff_model = None
        self.res_model = None
        self.xgb_model = None
        self.index_to_label = None
        
        # 특성 추출기 캐싱 (매번 생성하지 않음)
        self.eff_feature_extractor = None
        self.res_feature_extractor = None
        
        self.load_models()
    
    def build_transfer_model(self, base_cls, num_classes, dropout_rate=0.2):
        """전이학습 모델 구조 빌드 (가중치 로드용)"""
        base = base_cls(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        x = GlobalAveragePooling2D(name='gap')(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        out = Dense(num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=base.input, outputs=out)

    def load_models(self):
        """훈련된 모델들 로드"""
        print("훈련된 모델 로드 중...")
        
        try:
            # 라벨 매핑 로드
            label_map_path = os.path.join(self.model_dir, f"label_to_index_{self.timestamp}.joblib")
            label_map = joblib.load(label_map_path)
            self.index_to_label = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)
            print(f"클래스 수: {num_classes}")

            # EfficientNet 모델 로드
            self.eff_model = self.build_transfer_model(EfficientNetB0, num_classes, dropout_rate=0.2)
            eff_weights_path = os.path.join(self.model_dir, f"effnet_model_best_{self.timestamp}.h5")
            self.eff_model.load_weights(eff_weights_path)
            print("EfficientNet 모델 로드 완료")
            
            # 특성 추출기 미리 생성 (성능 최적화)
            self.eff_feature_extractor = Model(
                self.eff_model.input, 
                self.eff_model.get_layer('gap').output
            )

            # ResNet 모델 로드
            self.res_model = self.build_transfer_model(ResNet50V2, num_classes, dropout_rate=0.2)
            res_weights_path = os.path.join(self.model_dir, f"resnet_model_best_{self.timestamp}.h5")
            self.res_model.load_weights(res_weights_path)
            print("ResNet 모델 로드 완료")
            
            # 특성 추출기 미리 생성
            self.res_feature_extractor = Model(
                self.res_model.input, 
                self.res_model.get_layer('gap').output
            )

            # XGBoost 모델 로드
            xgb_path = os.path.join(self.model_dir, f"xgb_model_{self.timestamp}.joblib")
            self.xgb_model = joblib.load(xgb_path)
            print("XGBoost 모델 로드 완료")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise

    def download_image_from_url(self, url, timeout=3, max_retries=1):  # 더 빠르게 설정
        """URL에서 이미지 다운로드 - 최고속 버전"""
        
        for attempt in range(max_retries):
            try:
                # 최적화된 세션 사용
                response = session.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # 이미지 파일인지 빠른 확인
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    response.close()
                    return None, "이미지아님"
                
                # 파일 크기 확인 (5MB로 더 작게 제한)
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB 제한
                    response.close()
                    return None, "크기초과"
                
                # 이미지 데이터 읽기 (더 큰 청크로 빠르게)
                image_data = BytesIO()
                for chunk in response.iter_content(chunk_size=16384):  # 16KB 청크
                    image_data.write(chunk)
                image_data.seek(0)
                
                # PIL Image로 변환
                img = Image.open(image_data)
                
                # 크기가 너무 작은 이미지는 제외
                if img.width < 50 or img.height < 50:  # 조건 완화
                    return None, "크기작음"
                
                # RGB로 변환
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                return img, "성공"
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # 재시도 대기시간 더 단축
                    continue
                return None, f"네트워크오류"
            except Exception as e:
                return None, f"처리오류"
        
        return None, "재시도초과"

    def predict_single_image(self, pil_image):
        """단일 이미지 예측 - 성능 최적화"""
        try:
            # 이미지 전처리
            img_array = np.array(pil_image)
            img = tf.convert_to_tensor(img_array, dtype=tf.float32)
            img = tf.image.resize(img, IMG_SIZE)
            
            # 배치 차원 추가를 한 번만
            img_batch = tf.expand_dims(img, axis=0)
            
            # 각 모델용 전처리
            inp_eff = eff_pre(img_batch)
            inp_res = res_pre(img_batch)

            # CNN 모델들 예측 (배치로 처리)
            p_eff = self.eff_model.predict(inp_eff, verbose=0)
            p_res = self.res_model.predict(inp_res, verbose=0)
            p_cnn = (p_eff + p_res) / 2.0

            # XGBoost용 특성 추출 (미리 생성된 추출기 사용)
            feat_eff = self.eff_feature_extractor.predict(inp_eff, verbose=0)
            feat_res = self.res_feature_extractor.predict(inp_res, verbose=0)
            feat = np.hstack([feat_eff, feat_res])
            p_xgb = self.xgb_model.predict_proba(feat)

            # 앙상블 가중 평균 (CNN 60% + XGBoost 40%)
            ensemble = p_cnn * 0.6 + p_xgb * 0.4
            ensemble = ensemble.flatten()

            # 결과 추출
            idx = int(np.argmax(ensemble))
            predicted_label = self.index_to_label[idx]
            confidence = float(ensemble[idx])
            
            # 상위 3개 예측 결과 (벡터화 연산 사용)
            top3_indices = np.argsort(ensemble)[-3:][::-1]
            top3_results = [
                {
                    'label': self.index_to_label[int(i)],
                    'confidence': float(ensemble[i])
                }
                for i in top3_indices
            ]

            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'top3_predictions': top3_results,
                'status': '예측성공'
            }
            
        except Exception as e:
            return {
                'predicted_label': '예측실패',
                'confidence': 0.0,
                'top3_predictions': [],
                'status': f'예측오류: {str(e)}'
            }

def process_excel_file(excel_path, predictor, url_column_keywords=['url', 'URL', 'image', '이미지']):
    """단일 Excel 파일 처리"""
    print(f"\n처리 중: {os.path.basename(excel_path)}")
    
    try:
        # Excel 파일 읽기
        xl_file = pd.ExcelFile(excel_path)
        sheet_names = xl_file.sheet_names
        print(f"발견된 시트: {sheet_names}")
        
        results_summary = {}
        
        # 각 시트 처리
        for sheet_name in sheet_names:
            if '요약' in sheet_name or 'summary' in sheet_name.lower():
                continue  # 요약 시트는 건너뛰기
            
            print(f"\n  시트 처리: {sheet_name}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # URL 컬럼 찾기
            url_columns = []
            for col in df.columns:
                for keyword in url_column_keywords:
                    if keyword in col:
                        url_columns.append(col)
                        break
            
            if not url_columns:
                print(f"    URL 컬럼을 찾을 수 없습니다.")
                continue
            
            url_col = url_columns[0]  # 첫 번째 URL 컬럼 사용
            print(f"    사용할 URL 컬럼: {url_col}")
            
            # 결과 컬럼들 추가
            result_columns = ['예측_음식명', '예측_신뢰도', '2위_음식명', '2위_신뢰도', 
                            '3위_음식명', '3위_신뢰도', '처리상태', '처리시간']
            
            for col in result_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # 통계 초기화
            stats = {
                '총_처리수': 0,
                '성공수': 0,
                '실패수': 0,
                '예측결과_집계': {},
                '오류_종류': {}
            }
            
            # 각 URL 처리
            print(f"    총 {len(df)}개 URL 처리 시작...")
            
            for idx, row in df.iterrows():
                url = row[url_col]
                
                if pd.isna(url) or str(url).strip() == '':
                    df.loc[idx, '처리상태'] = 'URL없음'
                    stats['실패수'] += 1
                    print(f"    [{idx+1:3d}/{len(df)}] URL없음")
                    continue
                
                stats['총_처리수'] += 1
                
                start_time = time.time()
                
                # 이미지 다운로드
                pil_image, download_msg = predictor.download_image_from_url(url)
                
                if pil_image is None:
                    df.loc[idx, '처리상태'] = f'다운로드실패: {download_msg}'
                    stats['실패수'] += 1
                    if download_msg in stats['오류_종류']:
                        stats['오류_종류'][download_msg] += 1
                    else:
                        stats['오류_종류'][download_msg] = 1
                    
                    processing_time = round(time.time() - start_time, 2)
                    print(f"    [{idx+1:3d}/{len(df)}] 실패: {download_msg} ({processing_time:.1f}초)")
                    continue
                
                # 이미지 예측
                prediction_result = predictor.predict_single_image(pil_image)
                processing_time = round(time.time() - start_time, 2)
                
                # 결과 저장
                df.loc[idx, '예측_음식명'] = prediction_result['predicted_label']
                df.loc[idx, '예측_신뢰도'] = round(prediction_result['confidence'], 4)
                df.loc[idx, '처리상태'] = prediction_result['status']
                df.loc[idx, '처리시간'] = f"{processing_time}초"
                
                # 상위 3개 결과 저장
                top3 = prediction_result['top3_predictions']
                if len(top3) >= 2:
                    df.loc[idx, '2위_음식명'] = top3[1]['label']
                    df.loc[idx, '2위_신뢰도'] = round(top3[1]['confidence'], 4)
                if len(top3) >= 3:
                    df.loc[idx, '3위_음식명'] = top3[2]['label']
                    df.loc[idx, '3위_신뢰도'] = round(top3[2]['confidence'], 4)
                
                # 통계 업데이트 및 결과 출력
                if prediction_result['status'] == '예측성공':
                    stats['성공수'] += 1
                    predicted_food = prediction_result['predicted_label']
                    if predicted_food in stats['예측결과_집계']:
                        stats['예측결과_집계'][predicted_food] += 1
                    else:
                        stats['예측결과_집계'][predicted_food] = 1
                    
                    # 성공 결과 출력 (신뢰도와 함께)
                    confidence = prediction_result['confidence']
                    print(f"    [{idx+1:3d}/{len(df)}] 성공: {predicted_food} (신뢰도: {confidence:.3f}, {processing_time:.1f}초)")
                else:
                    stats['실패수'] += 1
                    error_msg = prediction_result['status']
                    if error_msg in stats['오류_종류']:
                        stats['오류_종류'][error_msg] += 1
                    else:
                        stats['오류_종류'][error_msg] = 1
                    
                    print(f"    [{idx+1:3d}/{len(df)}] 예측실패: {error_msg} ({processing_time:.1f}초)")
                
                # 10개마다 중간 통계 출력
                if (idx + 1) % 10 == 0:
                    current_success_rate = (stats['성공수'] / stats['총_처리수'] * 100) if stats['총_처리수'] > 0 else 0
                    print(f"    --- 중간 결과: {stats['총_처리수']}개 처리, 성공률 {current_success_rate:.1f}% ---")
                
                # 서버 부하 방지를 위한 최소 대기 (거의 없음)
                # time.sleep(0.01)  # 0.05초에서 0.01초로 더 단축 (거의 없음)
            
            # 수정된 시트 저장
            with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            results_summary[sheet_name] = stats
            
            success_rate = (stats['성공수'] / stats['총_처리수'] * 100) if stats['총_처리수'] > 0 else 0
            print(f"    완료: 총 {stats['총_처리수']}개, 성공 {stats['성공수']}개 ({success_rate:.1f}%), 실패 {stats['실패수']}개")
        
        # 요약 시트 생성
        create_summary_sheet(excel_path, results_summary)
        
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

def create_summary_sheet(excel_path, results_summary):
    """처리 결과 요약 시트 생성"""
    try:
        summary_data = []
        
        for sheet_name, stats in results_summary.items():
            # 기본 통계
            total = stats['총_처리수']
            success = stats['성공수']
            fail = stats['실패수']
            success_rate = (success / total * 100) if total > 0 else 0
            
            row_data = {
                '시트명': sheet_name,
                '총_처리수': total,
                '성공수': success,
                '실패수': fail,
                '성공률(%)': round(success_rate, 1),
                '처리일시': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 상위 예측 결과
            top_predictions = sorted(stats['예측결과_집계'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            
            for i, (food, count) in enumerate(top_predictions, 1):
                row_data[f'상위{i}_음식명'] = food
                row_data[f'상위{i}_개수'] = count
                row_data[f'상위{i}_비율(%)'] = round(count/success*100, 1) if success > 0 else 0
            
            # 주요 오류 유형
            top_errors = sorted(stats['오류_종류'].items(), 
                              key=lambda x: x[1], reverse=True)[:3]
            
            for i, (error, count) in enumerate(top_errors, 1):
                row_data[f'오류{i}_유형'] = error[:50]  # 길이 제한
                row_data[f'오류{i}_개수'] = count
            
            summary_data.append(row_data)
        
        # 전체 요약 추가
        total_processed = sum(s['총_처리수'] for s in results_summary.values())
        total_success = sum(s['성공수'] for s in results_summary.values())
        total_fail = sum(s['실패수'] for s in results_summary.values())
        overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
        
        overall_row = {
            '시트명': '전체_요약',
            '총_처리수': total_processed,
            '성공수': total_success,
            '실패수': total_fail,
            '성공률(%)': round(overall_success_rate, 1),
            '처리일시': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 전체 예측 결과 집계
        all_predictions = {}
        for stats in results_summary.values():
            for food, count in stats['예측결과_집계'].items():
                if food in all_predictions:
                    all_predictions[food] += count
                else:
                    all_predictions[food] = count
        
        top_overall = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (food, count) in enumerate(top_overall, 1):
            overall_row[f'상위{i}_음식명'] = food
            overall_row[f'상위{i}_개수'] = count
            overall_row[f'상위{i}_비율(%)'] = round(count/total_success*100, 1) if total_success > 0 else 0
        
        summary_data.append(overall_row)
        
        # DataFrame 생성 및 저장
        summary_df = pd.DataFrame(summary_data)
        
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='예측결과_요약', index=False)
        
        print(f"\n요약 시트 '예측결과_요약' 생성 완료!")
        print(f"전체 처리 결과: {total_processed}개 중 {total_success}개 성공 ({overall_success_rate:.1f}%)")
        
    except Exception as e:
        print(f"요약 시트 생성 중 오류: {e}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("음식 이미지 URL 예측 시스템")
    print("=" * 80)
    
    # ═══ 설정 변수 ═══════════════════════════════════════════
    EXCEL_FOLDER = r'C:\ai_x\source\Pks_Develop\N시기별음식URL수집\N월별조회메뉴수집_비중반영'
    MODEL_DIR = 'models'
    TIMESTAMP = '20250716_144252'
    # ════════════════════════════════════════════════════════
    
    # 폴더 및 모델 존재 확인
    if not os.path.exists(EXCEL_FOLDER):
        print(f"Excel 파일 폴더가 존재하지 않습니다: {EXCEL_FOLDER}")
        return
    
    if not os.path.exists(MODEL_DIR):
        print(f"모델 폴더가 존재하지 않습니다: {MODEL_DIR}")
        return
    
    # Excel 파일 목록 확인
    excel_files = glob.glob(os.path.join(EXCEL_FOLDER, "*.xlsx"))
    if not excel_files:
        print(f"처리할 Excel 파일이 없습니다: {EXCEL_FOLDER}")
        return
    
    print(f"발견된 Excel 파일: {len(excel_files)}개")
    for i, file in enumerate(excel_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # 예측기 초기화
    try:
        predictor = FoodImagePredictor(MODEL_DIR, TIMESTAMP)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 각 Excel 파일 처리
    print(f"\nExcel 파일 처리 시작...")
    start_time = time.time()
    
    for i, excel_file in enumerate(excel_files, 1):
        print(f"\n[{i}/{len(excel_files)}] 파일 처리 시작")
        process_excel_file(excel_file, predictor)
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"모든 처리가 완료되었습니다!")
    print(f"총 소요 시간: {total_time:.1f}초")
    print(f"결과 파일 위치: {EXCEL_FOLDER}")
    print("=" * 80)

if __name__ == '__main__':
    main()