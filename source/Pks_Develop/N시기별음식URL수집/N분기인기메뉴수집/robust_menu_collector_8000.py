import pandas as pd
import requests
import time
import random
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import logging
from urllib.parse import quote
import re
import warnings
import threading
from collections import deque
warnings.filterwarnings('ignore')

class RobustMenuImageCollector:
    def __init__(self, search_year=2024, search_quarter=1, target_images=8000):
        # 환경변수 강화된 검증
        self._validate_environment()
        
        self.search_year = search_year
        self.search_quarter = search_quarter
        self.search_months = self.get_quarter_months(search_quarter)
        self.target_images = target_images
        
        # API 엔드포인트들
        self.api_endpoints = {
            'news': "https://openapi.naver.com/v1/search/news.json",
            'blog': "https://openapi.naver.com/v1/search/blog.json", 
            'cafe': "https://openapi.naver.com/v1/search/cafearticle.json",
            'image': "https://openapi.naver.com/v1/search/image"
        }
        
        self.headers = {
            'X-Naver-Client-Id': self.client_id,
            'X-Naver-Client-Secret': self.client_secret,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 할당 비율 (기본 20% + 비례 80%)
        self.base_allocation_ratio = 0.2
        self.proportional_allocation_ratio = 0.8
        
        # 수학적 균등 가중치
        self.api_weights = {
            'news': 0.82,   
            'blog': 0.04,   
            'cafe': 0.14    
        }
        
        # API 제한 설정
        self.api_limits = {
            'max_requests_per_day': 25000,
            'max_start_position': 1000,  # 네이버 API 제한
            'retry_count': 3,
            'base_delay': 0.1,
            'retry_delay_base': 2
        }
        
        # 결과 저장
        self.popularity_results = []
        self.menu_quotas = {}
        self.collected_images = []
        self.collected_urls = set()  # 중복 방지
        self.failed_requests = []  # 실패한 요청 추적
        
        self.daily_request_count = 0
        self.current_collected_count = 0
        
        # 실시간 모니터링 변수들
        self.start_time = datetime.now()
        self.phase_start_time = datetime.now()
        self.current_phase = "초기화"
        self.total_phases = 3
        self.current_phase_num = 0
        self.phase_progress = 0.0
        self.overall_progress = 0.0
        self.recent_activities = deque(maxlen=10)  # 최근 10개 활동
        self.api_request_history = deque(maxlen=100)  # API 요청 히스토리
        self.collection_rate_history = deque(maxlen=20)  # 수집 속도 히스토리
        
        # 통계 정보
        self.stats = {
            'successful_analyses': 0,
            'failed_analyses': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'avg_analysis_time': 0.0,
            'avg_collection_time': 0.0,
            'estimated_completion_time': None
        }
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        print(f"견고한 메뉴 이미지 수집 시스템 초기화 완료")
        print(f"분석 기간: {search_year}년 {search_quarter}분기")
        print(f"목표 이미지: {target_images:,}개")
        print(f"할당 비율: 기본 {self.base_allocation_ratio*100}% + 비례 {self.proportional_allocation_ratio*100}%")
        
        # 초기 모니터링 정보 추가
        self._add_activity("시스템 초기화 완료")
        self._display_current_status()
    
    def _validate_environment(self):
        """환경변수 강화된 검증"""
        load_dotenv()
        
        self.client_id = os.getenv('Client_ID')
        self.client_secret = os.getenv('Client_Secret')
        
        # 상세한 환경변수 검증
        if not self.client_id:
            raise ValueError(
                "Client_ID가 설정되지 않았습니다.\n"
                ".env 파일에 다음과 같이 설정해주세요:\n"
                "Client_ID=your_actual_client_id\n"
                "Client_Secret=your_actual_client_secret"
            )
        
        if not self.client_secret:
            raise ValueError(
                "Client_Secret이 설정되지 않았습니다.\n"
                ".env 파일에 다음과 같이 설정해주세요:\n"
                "Client_ID=your_actual_client_id\n"
                "Client_Secret=your_actual_client_secret"
            )
        
        # API 키 형식 기본 검증
        if len(self.client_id) < 10 or len(self.client_secret) < 10:
            self.logger.warning("API 키가 너무 짧습니다. 올바른 키인지 확인해주세요.")
        
        print("환경변수 검증 완료")
    
    def _add_activity(self, activity_text):
        """최근 활동 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.recent_activities.append(f"[{timestamp}] {activity_text}")
    
    def _update_phase(self, phase_name, phase_num):
        """현재 단계 업데이트"""
        self.current_phase = phase_name
        self.current_phase_num = phase_num
        self.phase_start_time = datetime.now()
        self.phase_progress = 0.0
        self._add_activity(f"{phase_name} 시작")
    
    def _calculate_estimated_completion(self):
        """완료 예상 시간 계산"""
        if self.overall_progress > 0:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            total_estimated_time = elapsed_time / (self.overall_progress / 100)
            remaining_time = total_estimated_time - elapsed_time
            
            if remaining_time > 0:
                completion_time = datetime.now() + timedelta(seconds=remaining_time)
                self.stats['estimated_completion_time'] = completion_time
                return completion_time
        
        return None
    
    def _display_current_status(self):
        """현재 상태 실시간 표시"""
        elapsed_time = datetime.now() - self.start_time
        phase_elapsed = datetime.now() - self.phase_start_time
        
        print("\n" + "="*80)
        print("                      실시간 진행 상황 모니터링")
        print("="*80)
        
        # 전체 진행률
        print(f"전체 진행률: [{self._create_progress_bar(self.overall_progress, 40)}] {self.overall_progress:.1f}%")
        
        # 현재 단계 정보
        print(f"현재 단계: {self.current_phase} ({self.current_phase_num}/{self.total_phases})")
        print(f"단계 진행률: [{self._create_progress_bar(self.phase_progress, 30)}] {self.phase_progress:.1f}%")
        
        # 시간 정보
        print(f"경과 시간: {str(elapsed_time).split('.')[0]}")
        print(f"단계 경과: {str(phase_elapsed).split('.')[0]}")
        
        # 예상 완료 시간
        estimated_completion = self._calculate_estimated_completion()
        if estimated_completion:
            remaining = estimated_completion - datetime.now()
            print(f"예상 완료: {estimated_completion.strftime('%H:%M:%S')} (남은 시간: {str(remaining).split('.')[0]})")
        
        # 수집 통계
        print(f"\n수집 현황:")
        print(f"  이미지 수집: {self.current_collected_count:,}/{self.target_images:,}개 ({self.current_collected_count/self.target_images*100:.1f}%)")
        print(f"  API 요청: {self.daily_request_count:,}회 / {self.api_limits['max_requests_per_day']:,}회")
        print(f"  실패한 요청: {len(self.failed_requests)}회")
        
        # 성공률 통계
        if self.stats['successful_analyses'] + self.stats['failed_analyses'] > 0:
            analysis_success_rate = self.stats['successful_analyses'] / (self.stats['successful_analyses'] + self.stats['failed_analyses']) * 100
            print(f"  분석 성공률: {analysis_success_rate:.1f}%")
        
        if self.stats['successful_collections'] + self.stats['failed_collections'] > 0:
            collection_success_rate = self.stats['successful_collections'] / (self.stats['successful_collections'] + self.stats['failed_collections']) * 100
            print(f"  수집 성공률: {collection_success_rate:.1f}%")
        
        # 최근 활동
        print(f"\n최근 활동:")
        for activity in list(self.recent_activities)[-5:]:  # 최근 5개만 표시
            print(f"  {activity}")
        
        # 수집 속도 (최근 수집량 기반)
        if len(self.collection_rate_history) > 1:
            recent_rate = sum(self.collection_rate_history) / len(self.collection_rate_history)
            print(f"\n수집 속도: {recent_rate:.1f}개/분")
        
        print("="*80)
    
    def _create_progress_bar(self, percentage, length=30):
        """진행률 바 생성"""
        filled_length = int(length * percentage // 100)
        bar = '█' * filled_length + '░' * (length - filled_length)
        return bar
    
    def _record_api_request(self, api_type, success=True):
        """API 요청 기록"""
        self.api_request_history.append({
            'timestamp': datetime.now(),
            'api_type': api_type,
            'success': success
        })
    
    def _update_collection_rate(self, collected_count):
        """수집 속도 업데이트"""
        current_time = datetime.now()
        self.collection_rate_history.append(collected_count)
    
    def get_quarter_months(self, quarter):
        """분기별 월 정보 반환"""
        quarter_map = {
            1: [1, 2, 3], 2: [4, 5, 6], 
            3: [7, 8, 9], 4: [10, 11, 12]
        }
        return quarter_map.get(quarter, [1, 2, 3])
    
    def robust_api_call(self, api_type, query, start=1, display=100):
        """재시도 로직이 포함된 견고한 API 호출"""
        
        # API 제한 체크
        if self.daily_request_count >= self.api_limits['max_requests_per_day']:
            self.logger.error("일일 API 요청 한도에 도달했습니다.")
            self._record_api_request(api_type, False)
            return None
        
        # start 위치 제한 체크 (네이버 API 제한)
        if start > self.api_limits['max_start_position']:
            self.logger.warning(f"start 위치가 {self.api_limits['max_start_position']}을 초과했습니다.")
            self._record_api_request(api_type, False)
            return None
        
        api_url = self.api_endpoints.get(api_type)
        if not api_url:
            self.logger.error(f"지원하지 않는 API 타입: {api_type}")
            self._record_api_request(api_type, False)
            return None
        
        params = {
            'query': query,
            'start': start,
            'display': min(display, 100),
            'sort': 'date'
        }
        
        # 재시도 로직
        for attempt in range(self.api_limits['retry_count']):
            try:
                response = requests.get(
                    api_url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=15
                )
                self.daily_request_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    if 'items' in result:
                        self._record_api_request(api_type, True)
                        return result
                    else:
                        self.logger.warning(f"응답에 items가 없습니다: {api_type} - {query}")
                        self._record_api_request(api_type, True)
                        return {'items': []}
                
                elif response.status_code == 429:
                    # Rate limit - 더 긴 대기
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20초
                    self.logger.warning(f"API 제한 도달. {wait_time}초 대기... (시도 {attempt + 1}/{self.api_limits['retry_count']})")
                    self._add_activity(f"API 제한으로 {wait_time}초 대기")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code in [400, 401, 403]:
                    # 인증 오류 - 재시도 불가
                    self.logger.error(f"API 인증 오류 ({response.status_code}): {response.text}")
                    self._record_api_request(api_type, False)
                    return None
                
                else:
                    # 기타 오류 - 재시도
                    self.logger.warning(f"API 오류 ({response.status_code}). 재시도 {attempt + 1}/{self.api_limits['retry_count']}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"API 타임아웃. 재시도 {attempt + 1}/{self.api_limits['retry_count']}")
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"네트워크 오류: {e}. 재시도 {attempt + 1}/{self.api_limits['retry_count']}")
            
            # 재시도 전 대기 (지수 백오프)
            if attempt < self.api_limits['retry_count'] - 1:
                wait_time = (self.api_limits['retry_delay_base'] ** attempt) * random.uniform(0.5, 1.5)
                time.sleep(wait_time)
        
        # 모든 재시도 실패
        self.failed_requests.append({
            'api_type': api_type,
            'query': query,
            'start': start,
            'timestamp': datetime.now().isoformat()
        })
        self.logger.error(f"API 호출 최종 실패: {api_type} - {query}")
        self._record_api_request(api_type, False)
        return None
    
    def strict_date_validation(self, date_str, api_type):
        """엄격한 날짜 검증"""
        if not date_str or not isinstance(date_str, str):
            return False
        
        try:
            if api_type == 'news':
                # 뉴스: "Mon, 15 Jun 2024 09:30:00 +0900" 형식
                from email.utils import parsedate_tz
                parsed = parsedate_tz(date_str.strip())
                if parsed:
                    dt = datetime(*parsed[:6])
                    is_valid = dt.year == self.search_year and dt.month in self.search_months
                    return is_valid
            
            elif api_type in ['blog', 'cafe']:
                # 블로그/카페: "20240615" 형식
                if len(date_str) >= 8 and date_str.isdigit():
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    is_valid = year == self.search_year and month in self.search_months
                    return is_valid
            
            elif api_type == 'image':
                # 이미지: pubDate 또는 URL 패턴 분석
                # 먼저 pubDate 시도
                if self.strict_date_validation(date_str, 'news'):
                    return True
                
                # URL 패턴 분석 (보조적)
                return self.extract_date_from_url(date_str)
            
        except (ValueError, IndexError, TypeError) as e:
            self.logger.debug(f"날짜 파싱 실패: {date_str} - {e}")
            return False
        
        return False
    
    def extract_date_from_url(self, url):
        """URL에서 날짜 추출 (보조적 방법)"""
        if not url:
            return False
        
        try:
            # 다양한 날짜 패턴
            patterns = [
                r'(\d{4})/(\d{1,2})/(\d{1,2})',  # 2024/1/15
                r'(\d{4})-(\d{1,2})-(\d{1,2})',  # 2024-1-15
                r'(\d{4})(\d{2})(\d{2})',        # 20240115
                r'(\d{4})/(\d{1,2})',            # 2024/1
                r'(\d{4})-(\d{1,2})',            # 2024-1
                r'(\d{4})(\d{2})',               # 202401
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    
                    if year == self.search_year and month in self.search_months:
                        return True
            
            return False
            
        except (ValueError, IndexError, AttributeError):
            return False
    
    def load_menu_data(self, csv_file_path):
        """메뉴 데이터 로드 with 검증"""
        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            # 필수 컬럼 검증
            required_columns = ['대분류', '중분류', '소분류', '상세메뉴', '시각적특징']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
            
            print(f"CSV 파일 로드 완료: {len(df)}개 행")
            self._add_activity(f"CSV 파일 로드 완료: {len(df)}개 행")
            
            menu_items = []
            for _, row in df.iterrows():
                detail_menus = [menu.strip() for menu in str(row['상세메뉴']).split(',') if menu.strip()]
                for detail_menu in detail_menus:
                    if detail_menu:  # 빈 문자열 제외
                        menu_items.append({
                            '대분류': row['대분류'],
                            '중분류': row['중분류'],
                            '소분류': row['소분류'],
                            '상세메뉴': detail_menu,
                            '시각적특징': row['시각적특징']
                        })
            
            print(f"총 {len(menu_items)}개의 개별 메뉴 항목 생성")
            self._add_activity(f"메뉴 항목 생성: {len(menu_items)}개")
            return menu_items
            
        except Exception as e:
            self.logger.error(f"메뉴 데이터 로드 실패: {e}")
            raise
    
    def count_period_content_robust(self, api_type, menu_name):
        """견고한 콘텐츠 수 계산 (중복 제거 포함)"""
        unique_content_urls = set()  # 중복 제거용
        search_keywords = self.generate_api_keywords(menu_name, api_type)
        
        for keyword in search_keywords:
            start = 1
            max_pages = 3  # 8000개 목표에 맞춰 페이지 증가
            
            for page in range(max_pages):
                # start 위치 제한 체크
                if start > self.api_limits['max_start_position']:
                    break
                
                result = self.robust_api_call(api_type, keyword, start=start, display=100)
                
                if not result or 'items' not in result:
                    break
                
                items = result['items']
                if not items:
                    break
                
                valid_count_in_page = 0
                
                for item in items:
                    # API별 날짜 필드 추출
                    if api_type == 'news':
                        date_field = item.get('pubDate', '')
                        content_url = item.get('originallink', '') or item.get('link', '')
                    else:
                        date_field = item.get('postdate', '')
                        content_url = item.get('link', '')
                    
                    # 날짜 검증 및 중복 체크
                    if (self.strict_date_validation(date_field, api_type) and 
                        content_url and 
                        content_url not in unique_content_urls):
                        
                        unique_content_urls.add(content_url)
                        valid_count_in_page += 1
                
                # 해당 분기 콘텐츠가 없으면 중단 (날짜순 정렬)
                if valid_count_in_page == 0:
                    break
                
                start += len(items)
                time.sleep(self.api_limits['base_delay'])
        
        return len(unique_content_urls)
    
    def generate_api_keywords(self, menu_name, api_type):
        """API별 최적화된 검색 키워드 생성"""
        base_keywords = {
            'news': [menu_name, f"{menu_name} 맛집"],
            'blog': [menu_name, f"{menu_name} 후기"],
            'cafe': [menu_name, f"{menu_name} 추천"],
            'image': [menu_name, f"{menu_name} 음식", f"{menu_name} 요리"]
        }
        
        return base_keywords.get(api_type, [menu_name])
    
    def analyze_menu_popularity_robust(self, menu_item):
        """견고한 메뉴 인기도 분석"""
        menu_name = menu_item['상세메뉴']
        analysis_start = datetime.now()
        
        try:
            # 각 API에서 콘텐츠 수 계산
            news_count = self.count_period_content_robust('news', menu_name)
            blog_count = self.count_period_content_robust('blog', menu_name)
            cafe_count = self.count_period_content_robust('cafe', menu_name)
            
            # 수학적 균등 가중치 적용
            balanced_score = (
                news_count * self.api_weights['news'] +
                blog_count * self.api_weights['blog'] +
                cafe_count * self.api_weights['cafe']
            )
            
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            self.stats['avg_analysis_time'] = (
                (self.stats['avg_analysis_time'] * self.stats['successful_analyses'] + analysis_time) / 
                (self.stats['successful_analyses'] + 1)
            )
            self.stats['successful_analyses'] += 1
            
            return {
                'menu_name': menu_name,
                'news_count': news_count,
                'blog_count': blog_count,
                'cafe_count': cafe_count,
                'balanced_score': round(balanced_score, 2),
                'analysis_success': True,
                'analysis_time': analysis_time
            }
            
        except Exception as e:
            self.logger.error(f"메뉴 '{menu_name}' 인기도 분석 실패: {e}")
            self.stats['failed_analyses'] += 1
            
            return {
                'menu_name': menu_name,
                'news_count': 0,
                'blog_count': 0,
                'cafe_count': 0,
                'balanced_score': 0.0,
                'analysis_success': False,
                'error_message': str(e)
            }
    
    def step1_analyze_popularity(self, csv_file_path):
        """1단계: 견고한 인기도 분석"""
        self._update_phase("메뉴별 인기도 분석", 1)
        
        menu_items = self.load_menu_data(csv_file_path)
        total_menus = len(menu_items)
        
        for i, menu_item in enumerate(menu_items):
            menu_name = menu_item['상세메뉴']
            self._add_activity(f"분석 중: {menu_name}")
            
            popularity_result = self.analyze_menu_popularity_robust(menu_item)
            popularity_result.update({
                '대분류': menu_item['대분류'],
                '중분류': menu_item['중분류'],
                '소분류': menu_item['소분류'],
                '시각적특징': menu_item['시각적특징']
            })
            
            self.popularity_results.append(popularity_result)
            
            # 진행률 업데이트
            self.phase_progress = ((i + 1) / total_menus) * 100
            self.overall_progress = (self.phase_progress / self.total_phases) * (1/3)
            
            if popularity_result['analysis_success']:
                self._add_activity(f"분석 완료: {menu_name} (점수: {popularity_result['balanced_score']:.2f})")
            else:
                self._add_activity(f"분석 실패: {menu_name}")
            
            # 실시간 상태 표시 (10개마다)
            if (i + 1) % 10 == 0:
                self._display_current_status()
            
            time.sleep(random.uniform(0.2, 0.5))
        
        self._add_activity(f"1단계 완료: {self.stats['successful_analyses']}개 성공, {self.stats['failed_analyses']}개 실패")
        self._display_current_status()
        
        return menu_items
    
    def step2_calculate_quotas_precise(self, menu_items):
        """2단계: 정밀한 할당량 계산"""
        self._update_phase("할당량 계산", 2)
        
        # 성공한 분석 결과만 사용
        successful_results = [r for r in self.popularity_results if r['analysis_success']]
        total_balanced_score = sum([r['balanced_score'] for r in successful_results])
        
        self._add_activity(f"유효한 분석 결과: {len(successful_results)}개")
        self._add_activity(f"총 균등화 점수: {total_balanced_score:.2f}점")
        
        # 정밀한 할당량 계산
        base_total = int(self.target_images * self.base_allocation_ratio)
        base_quota_per_menu = max(1, base_total // len(menu_items))
        used_base_total = base_quota_per_menu * len(menu_items)
        proportional_total = self.target_images - used_base_total
        
        self._add_activity(f"기본 할당량: {used_base_total}개")
        self._add_activity(f"비례 할당량: {proportional_total}개")
        
        allocated_total = 0
        
        for result in self.popularity_results:
            menu_name = result['menu_name']
            balanced_score = result['balanced_score']
            
            # 기본 할당량
            base_quota = base_quota_per_menu
            
            # 비례 할당량 (성공한 경우만)
            if result['analysis_success'] and total_balanced_score > 0:
                ratio = balanced_score / total_balanced_score
                additional_quota = int(proportional_total * ratio)
            else:
                additional_quota = 0
            
            total_quota = base_quota + additional_quota
            self.menu_quotas[menu_name] = total_quota
            allocated_total += total_quota
        
        # 할당량 검증
        quota_difference = self.target_images - allocated_total
        if abs(quota_difference) > 0:
            self._add_activity(f"할당량 차이 조정: {quota_difference}개")
        
        # TOP 할당량 표시
        sorted_quotas = sorted(self.menu_quotas.items(), key=lambda x: x[1], reverse=True)
        top_menu = sorted_quotas[0] if sorted_quotas else ("없음", 0)
        self._add_activity(f"최고 할당량: {top_menu[0]} ({top_menu[1]}개)")
        
        self.phase_progress = 100.0
        self.overall_progress = ((1 + 1) / self.total_phases) * 100 / 3
        
        self._add_activity(f"2단계 완료: 총 할당량 {allocated_total:,}개")
        self._display_current_status()
    
    def collect_menu_images_robust(self, menu_name, target_quota):
        """견고한 메뉴별 이미지 수집"""
        if target_quota <= 0:
            return []
        
        collection_start = datetime.now()
        collected_images = []
        search_keywords = self.generate_api_keywords(menu_name, 'image')
        
        for keyword in search_keywords:
            if len(collected_images) >= target_quota:
                break
            
            start = 1
            max_pages = 5  # 8000개 목표에 맞춰 페이지 증가
            
            for page in range(max_pages):
                if len(collected_images) >= target_quota:
                    break
                
                # start 위치 제한 체크
                if start > self.api_limits['max_start_position']:
                    break
                
                result = self.robust_api_call('image', keyword, start=start, display=100)
                
                if not result or 'items' not in result:
                    break
                
                images = result['items']
                if not images:
                    break
                
                for img in images:
                    if len(collected_images) >= target_quota:
                        break
                    
                    img_url = img.get('link', '')
                    pub_date = img.get('pubDate', '')
                    
                    # 엄격한 검증: URL 중복 + 날짜 검증
                    if (img_url and 
                        img_url not in self.collected_urls and
                        (self.strict_date_validation(pub_date, 'image') or 
                         self.strict_date_validation(img_url, 'image'))):
                        
                        self.collected_urls.add(img_url)
                        
                        image_data = {
                            'menu_name': menu_name,
                            'image_url': img_url,
                            'title': img.get('title', '').replace('<b>', '').replace('</b>', ''),
                            'thumbnail': img.get('thumbnail', ''),
                            'size_width': img.get('sizewidth', ''),
                            'size_height': img.get('sizeheight', ''),
                            'pub_date': pub_date,
                            'search_keyword': keyword,
                            'collected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        collected_images.append(image_data)
                
                start += len(images)
                time.sleep(self.api_limits['base_delay'])
        
        # 수집 통계 업데이트
        if collected_images:
            collection_time = (datetime.now() - collection_start).total_seconds()
            self.stats['avg_collection_time'] = (
                (self.stats['avg_collection_time'] * self.stats['successful_collections'] + collection_time) / 
                (self.stats['successful_collections'] + 1)
            )
            self.stats['successful_collections'] += 1
            self._update_collection_rate(len(collected_images))
        else:
            self.stats['failed_collections'] += 1
        
        return collected_images
    
    def step3_collect_images_robust(self, menu_items):
        """3단계: 견고한 이미지 수집"""
        self._update_phase("이미지 수집", 3)
        
        sorted_quotas = sorted(self.menu_quotas.items(), key=lambda x: x[1], reverse=True)
        total_menus = len(sorted_quotas)
        
        for i, (menu_name, target_quota) in enumerate(sorted_quotas):
            if self.current_collected_count >= self.target_images:
                self._add_activity(f"목표 달성으로 수집 완료!")
                break
            
            self._add_activity(f"수집 중: {menu_name} (할당량: {target_quota}개)")
            
            remaining_quota = self.target_images - self.current_collected_count
            actual_quota = min(target_quota, remaining_quota)
            
            try:
                images = self.collect_menu_images_robust(menu_name, actual_quota)
                
                if images:
                    # 메뉴 정보 추가
                    menu_info = next((item for item in menu_items if item['상세메뉴'] == menu_name), {})
                    for img in images:
                        img.update({
                            '대분류': menu_info.get('대분류', ''),
                            '중분류': menu_info.get('중분류', ''),
                            '소분류': menu_info.get('소분류', ''),
                            '시각적특징': menu_info.get('시각적특징', ''),
                            '분석분기': f"{self.search_year}년 {self.search_quarter}분기"
                        })
                    
                    self.collected_images.extend(images)
                    self.current_collected_count += len(images)
                    
                    success_rate = (len(images) / target_quota * 100) if target_quota > 0 else 0
                    self._add_activity(f"수집 완료: {menu_name} ({len(images)}개, 달성률: {success_rate:.1f}%)")
                else:
                    self._add_activity(f"수집 실패: {menu_name}")
                
            except Exception as e:
                self.logger.error(f"메뉴 '{menu_name}' 이미지 수집 실패: {e}")
                self._add_activity(f"수집 오류: {menu_name} - {str(e)}")
            
            # 진행률 업데이트
            self.phase_progress = ((i + 1) / total_menus) * 100
            self.overall_progress = ((2 + self.phase_progress/100) / self.total_phases) * 100
            
            # 실시간 상태 표시 (20개마다)
            if (i + 1) % 20 == 0:
                self._display_current_status()
            
            time.sleep(random.uniform(0.2, 0.5))
        
        self._add_activity(f"3단계 완료: 총 {self.current_collected_count:,}개 수집")
        self.overall_progress = 100.0
        self._display_current_status()
    
    def save_robust_results(self, output_file="robust_menu_images_8000_2024Q1.xlsx"):
        """견고한 결과 저장"""
        try:
            self._add_activity("결과 저장 중...")
            
            # 인기도 분석 결과
            popularity_df = pd.DataFrame(self.popularity_results)
            popularity_df['할당량'] = popularity_df['menu_name'].map(self.menu_quotas)
            popularity_df = popularity_df.sort_values('balanced_score', ascending=False)
            
            # 이미지 수집 결과
            if self.collected_images:
                images_df = pd.DataFrame(self.collected_images)
                
                # 메뉴별 수집 실적
                collection_stats = images_df.groupby('menu_name').size().reset_index(name='실제수집량')
                collection_stats = collection_stats.merge(
                    popularity_df[['menu_name', '할당량', 'balanced_score']], 
                    on='menu_name', 
                    how='right'
                ).fillna(0)
                collection_stats['달성률'] = (collection_stats['실제수집량'] / collection_stats['할당량'] * 100).round(1)
                collection_stats = collection_stats.sort_values('실제수집량', ascending=False)
            else:
                images_df = pd.DataFrame()
                collection_stats = pd.DataFrame()
            
            # 실패한 요청 통계
            failed_requests_df = pd.DataFrame(self.failed_requests) if self.failed_requests else pd.DataFrame()
            
            # 실시간 모니터링 로그
            monitoring_log = list(self.recent_activities) if self.recent_activities else []
            monitoring_df = pd.DataFrame(monitoring_log, columns=['활동로그']) if monitoring_log else pd.DataFrame()
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 인기도 분석 결과
                popularity_df.to_excel(writer, sheet_name='인기도분석결과', index=False)
                
                # 이미지 수집 결과
                if not images_df.empty:
                    images_df.to_excel(writer, sheet_name='수집된이미지', index=False)
                
                # 메뉴별 수집 실적
                if not collection_stats.empty:
                    collection_stats.to_excel(writer, sheet_name='메뉴별수집실적', index=False)
                
                # 실패한 요청들
                if not failed_requests_df.empty:
                    failed_requests_df.to_excel(writer, sheet_name='실패한요청', index=False)
                
                # 모니터링 로그
                if not monitoring_df.empty:
                    monitoring_df.to_excel(writer, sheet_name='실행로그', index=False)
                
                # 전체 요약
                total_elapsed = datetime.now() - self.start_time
                months_str = f"{self.search_months[0]}-{self.search_months[-1]}월"
                
                summary_data = {
                    '항목': [
                        '분석 기간',
                        '총 메뉴 수',
                        '성공한 인기도 분석',
                        '실패한 인기도 분석',
                        '목표 이미지 수',
                        '실제 수집 이미지',
                        '목표 달성률',
                        '기본 할당 비율',
                        '비례 할당 비율',
                        'API 총 요청 횟수',
                        '실패한 API 요청',
                        '총 실행 시간',
                        '평균 분석 시간',
                        '평균 수집 시간',
                        '평균 메뉴당 수집',
                        '수집 완료 시각'
                    ],
                    '값': [
                        f"{self.search_year}년 {months_str}",
                        f"{len(self.popularity_results)}개",
                        f"{self.stats['successful_analyses']}개",
                        f"{self.stats['failed_analyses']}개",
                        f"{self.target_images:,}개",
                        f"{len(self.collected_images):,}개",
                        f"{(len(self.collected_images)/self.target_images*100):.1f}%",
                        f"{self.base_allocation_ratio*100}%",
                        f"{self.proportional_allocation_ratio*100}%",
                        f"{self.daily_request_count:,}회",
                        f"{len(self.failed_requests)}회",
                        f"{str(total_elapsed).split('.')[0]}",
                        f"{self.stats['avg_analysis_time']:.2f}초",
                        f"{self.stats['avg_collection_time']:.2f}초",
                        f"{len(self.collected_images)/len(self.popularity_results):.1f}개" if self.popularity_results else "0개",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='전체요약', index=False)
            
            self._add_activity(f"결과 저장 완료: {output_file}")
            print(f"\n견고한 결과 저장 완료: {output_file}")
            print(f"수집된 이미지: {len(self.collected_images):,}개")
            print(f"목표 달성률: {(len(self.collected_images)/self.target_images*100):.1f}%")
            print(f"총 실행 시간: {str(datetime.now() - self.start_time).split('.')[0]}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            print(f"결과 저장 중 오류 발생: {e}")
    
    def run_robust_process(self, csv_file_path):
        """견고한 통합 프로세스 실행"""
        try:
            print("견고한 메뉴 이미지 수집 프로세스 시작")
            print("="*80)
            self._add_activity("프로세스 시작")
            
            # 1단계: 견고한 인기도 분석
            menu_items = self.step1_analyze_popularity(csv_file_path)
            
            # 2단계: 정밀한 할당량 계산  
            self.step2_calculate_quotas_precise(menu_items)
            
            # 3단계: 견고한 이미지 수집
            self.step3_collect_images_robust(menu_items)
            
            # 결과 저장
            self.save_robust_results()
            
            # 최종 요약
            total_elapsed = datetime.now() - self.start_time
            print("\n" + "="*80)
            print("                    견고한 프로세스 완료!")
            print("="*80)
            print(f"최종 수집: {len(self.collected_images):,}개 이미지")
            print(f"목표 달성률: {(len(self.collected_images)/self.target_images*100):.1f}%")
            print(f"총 실행 시간: {str(total_elapsed).split('.')[0]}")
            print(f"API 사용량: {self.daily_request_count:,}회")
            print(f"실패한 요청: {len(self.failed_requests)}회")
            print(f"분석 성공률: {(self.stats['successful_analyses']/(self.stats['successful_analyses']+self.stats['failed_analyses'])*100):.1f}%")
            print(f"수집 성공률: {(self.stats['successful_collections']/(self.stats['successful_collections']+self.stats['failed_collections'])*100):.1f}%")
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"견고한 프로세스 실행 중 치명적 오류: {e}")
            print(f"프로세스 실행 중 치명적 오류 발생: {e}")
            
            # 부분 결과라도 저장 시도
            if self.collected_images:
                try:
                    self.save_robust_results("partial_results_8000.xlsx")
                    print("부분 결과를 저장했습니다: partial_results_8000.xlsx")
                except:
                    print("부분 결과 저장도 실패했습니다.")
            
            import traceback
            traceback.print_exc()


def main():
    """메인 실행 함수"""
    CSV_FILE_PATH = "식당대12중53소132상세메뉴379분류.csv"
    SEARCH_YEAR = 2024
    SEARCH_QUARTER = 1
    TARGET_IMAGES = 8000  # 목표량 8000개로 변경
    
    try:
        print("견고한 통합 메뉴 이미지 수집 시스템 (8000개 목표)")
        print("="*60)
        
        # 견고한 시스템 초기화
        collector = RobustMenuImageCollector(
            search_year=SEARCH_YEAR,
            search_quarter=SEARCH_QUARTER,
            target_images=TARGET_IMAGES
        )
        
        # 파일 존재 확인
        if not os.path.exists(CSV_FILE_PATH):
            print(f"오류: CSV 파일을 찾을 수 없습니다 - {CSV_FILE_PATH}")
            print("현재 디렉토리의 파일들:")
            for file in os.listdir('.'):
                if file.endswith('.csv'):
                    print(f"  - {file}")
            return
        
        # .env 파일 확인
        if not os.path.exists('.env'):
            print("오류: .env 파일이 필요합니다.")
            print("다음 내용으로 .env 파일을 생성해주세요:")
            print("Client_ID=your_client_id")
            print("Client_Secret=your_client_secret")
            return
        
        # 견고한 프로세스 실행
        collector.run_robust_process(CSV_FILE_PATH)
        
    except Exception as e:
        print(f"메인 함수에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()