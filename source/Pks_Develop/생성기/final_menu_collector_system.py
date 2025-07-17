"""
메뉴 이미지 수집 시스템 - 최종 완성 버전
- 0개 수집 메뉴 자동 재수집
- 연속 분기 작업 지원
- 중단 기능 추가
- 결과 파일 통합 관리
"""

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
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import deque
warnings.filterwarnings('ignore')


class MenuImageCollectorFinal:
    def __init__(self, search_year=2025, search_quarter=1, target_images=8000):
        """메뉴 이미지 수집 시스템 초기화"""
        # 환경변수 검증
        self._validate_environment()
        
        self.search_year = search_year
        self.search_quarter = search_quarter
        self.search_months = self.get_quarter_months(search_quarter)
        self.target_images = target_images
        
        # API 엔드포인트들 (카페 제외)
        self.api_endpoints = {
            'news': "https://openapi.naver.com/v1/search/news.json",
            'blog': "https://openapi.naver.com/v1/search/blog.json", 
            'image': "https://openapi.naver.com/v1/search/image"
        }
        
        self.headers = {
            'X-Naver-Client-Id': self.client_id,
            'X-Naver-Client-Secret': self.client_secret,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 할당 비율 설정
        self.base_allocation_ratio = 0.2
        self.proportional_allocation_ratio = 0.8
        
        # 기존 가중치에서 카페만 제외하고 정규화
        original_news_weight = 0.82
        original_blog_weight = 0.04
        total_remaining_weight = original_news_weight + original_blog_weight
        
        self.api_weights = {
            'news': round(original_news_weight / total_remaining_weight, 3),  # 0.953
            'blog': round(original_blog_weight / total_remaining_weight, 3)   # 0.047
        }
        
        # API 제한 설정
        self.api_limits = {
            'max_requests_per_day': 25000,
            'max_start_position': 1000,
            'retry_count': 3,
            'base_delay': 0.1,
            'retry_delay_base': 2
        }
        
        # 결과 저장 변수들
        self.popularity_results = []
        self.menu_quotas = {}
        self.collected_images = []
        self.collected_urls = set()
        self.failed_requests = []
        
        # 카운터 변수들
        self.daily_request_count = 0
        self.current_collected_count = 0
        
        # 재수집 관련 변수들
        self.retry_target_menus = []  # 재수집 대상 메뉴들
        self.retry_collected_count = 0  # 재수집으로 수집된 이미지 수
        self.is_retry_phase = False  # 재수집 단계 여부
        
        # 실시간 모니터링 변수들
        self.start_time = datetime.now()
        self.phase_start_time = datetime.now()
        self.current_phase = "대기"
        self.total_phases = 3
        self.current_phase_num = 0
        self.phase_progress = 0.0
        self.overall_progress = 0.0
        self.recent_activities = deque(maxlen=20)
        self.is_running = False
        self.is_stopped = False  # 중단 플래그
        
        # 통계 정보
        self.stats = {
            'successful_analyses': 0,
            'failed_analyses': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'news_content_count': 0,
            'blog_content_count': 0,
            'avg_analysis_time': 0.0,
            'avg_collection_time': 0.0,
            'top_menus': [],
            'retry_successful': 0,  # 재수집 성공
            'retry_failed': 0       # 재수집 실패
        }
        
        # 실제 콘텐츠 비율 추적
        self.actual_content_ratios = {
            'news_to_blog_ratio': None,
            'expected_ratio': original_news_weight / original_blog_weight,
            'weight_adjustment_needed': False
        }
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('menu_collector.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print(f"메뉴 이미지 수집 시스템 초기화 완료")
        print(f"분석 기간: {search_year}년 {search_quarter}분기")
        print(f"목표 이미지: {target_images:,}개")
        
        self._add_activity("시스템 초기화 완료")
    
    def _validate_environment(self):
        """환경변수 검증"""
        load_dotenv()
        
        self.client_id = os.getenv('Client_ID')
        self.client_secret = os.getenv('Client_Secret')
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "API 키가 설정되지 않았습니다.\n"
                ".env 파일에 다음과 같이 설정해주세요:\n"
                "Client_ID=your_client_id\n"
                "Client_Secret=your_client_secret"
            )
        
        print("환경변수 검증 완료")
    
    def _add_activity(self, activity_text):
        """최근 활동 로그에 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.recent_activities.append(f"[{timestamp}] {activity_text}")
        self.logger.info(activity_text)
    
    def _update_phase(self, phase_name, phase_num):
        """현재 단계 정보 업데이트"""
        self.current_phase = phase_name
        self.current_phase_num = phase_num
        self.phase_start_time = datetime.now()
        self.phase_progress = 0.0
        self._add_activity(f"{phase_name} 시작")
    
    def get_quarter_months(self, quarter):
        """분기별 해당 월 리스트 반환"""
        quarter_map = {
            1: [1, 2, 3], 2: [4, 5, 6], 
            3: [7, 8, 9], 4: [10, 11, 12]
        }
        return quarter_map.get(quarter, [1, 2, 3])
    
    def reset_for_new_period(self):
        """새로운 분기 작업을 위한 초기화"""
        self.popularity_results = []
        self.menu_quotas = {}
        self.collected_images = []
        self.collected_urls = set()
        self.failed_requests = []
        self.current_collected_count = 0
        self.retry_target_menus = []
        self.retry_collected_count = 0
        self.is_retry_phase = False
        self.is_running = False
        self.is_stopped = False
        self.overall_progress = 0.0
        self.phase_progress = 0.0
        self.current_phase = "대기"
        self.current_phase_num = 0
        
        # 통계 초기화
        self.stats = {
            'successful_analyses': 0,
            'failed_analyses': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'news_content_count': 0,
            'blog_content_count': 0,
            'avg_analysis_time': 0.0,
            'avg_collection_time': 0.0,
            'top_menus': [],
            'retry_successful': 0,
            'retry_failed': 0
        }
        
        self.start_time = datetime.now()
        self._add_activity(f"새 작업 초기화: {self.search_year}년 {self.search_quarter}분기")
    
    def stop_process(self):
        """프로세스 중단"""
        self.is_stopped = True
        self.is_running = False
        self._add_activity("사용자에 의해 중단됨")
    
    def robust_api_call(self, api_type, query, start=1, display=100):
        """재시도 로직이 포함된 견고한 API 호출"""
        if self.is_stopped:
            return None
            
        if self.daily_request_count >= self.api_limits['max_requests_per_day']:
            self.logger.error("일일 API 요청 한도에 도달했습니다.")
            return None
        
        if start > self.api_limits['max_start_position']:
            return None
        
        api_url = self.api_endpoints.get(api_type)
        if not api_url:
            return None
        
        sort_param = 'sim' if api_type == 'image' else 'date'
        
        params = {
            'query': query,
            'start': start,
            'display': min(display, 100),
            'sort': sort_param
        }
        
        for attempt in range(self.api_limits['retry_count']):
            if self.is_stopped:
                return None
                
            try:
                response = requests.get(api_url, headers=self.headers, params=params, timeout=15)
                self.daily_request_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    return result if 'items' in result else {'items': []}
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 5
                    self._add_activity(f"API 제한으로 {wait_time}초 대기")
                    time.sleep(wait_time)
                    continue
                else:
                    break
                    
            except Exception as e:
                if attempt == self.api_limits['retry_count'] - 1:
                    self.logger.warning(f"API 호출 실패: {e}")
                time.sleep(self.api_limits['retry_delay_base'] ** attempt)
        
        self.failed_requests.append({
            'api_type': api_type,
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
        return None
    
    def validate_blog_date(self, date_str):
        """개선된 블로그 postdate 검증 (완화)"""
        if not date_str or not isinstance(date_str, str):
            return True  # postdate 없으면 포함
        
        try:
            patterns = [
                r'(\d{4})(\d{2})(\d{2})',      # 20250215
                r'(\d{4})-(\d{2})-(\d{2})',    # 2025-02-15
                r'(\d{4})\.(\d{2})\.(\d{2})',  # 2025.02.15
                r'(\d{4})/(\d{2})/(\d{2})',    # 2025/02/15
                r'(\d{4})(\d{2})',             # 202502
            ]
            
            for pattern in patterns:
                match = re.match(pattern, date_str.strip())
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    return year == self.search_year and month in self.search_months
            
            return True  # 패턴 매칭 실패시에도 포함
            
        except:
            return True
    
    def validate_news_date(self, date_str):
        """뉴스 pubDate 검증"""
        if not date_str:
            return False
        
        try:
            from email.utils import parsedate_tz
            parsed = parsedate_tz(date_str.strip())
            if parsed:
                dt = datetime(*parsed[:6])
                return dt.year == self.search_year and dt.month in self.search_months
            return False
        except:
            return False
    
    def load_menu_data(self, csv_file_path):
        """메뉴 데이터 CSV 파일 로드"""
        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            required_columns = ['대분류', '중분류', '소분류', '상세메뉴', '시각적특징']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
            
            self._add_activity(f"CSV 파일 로드: {len(df)}개 행")
            
            menu_items = []
            for _, row in df.iterrows():
                detail_menus = [menu.strip() for menu in str(row['상세메뉴']).split(',') if menu.strip()]
                for detail_menu in detail_menus:
                    if detail_menu:
                        menu_items.append({
                            '대분류': row['대분류'],
                            '중분류': row['중분류'],
                            '소분류': row['소분류'],
                            '상세메뉴': detail_menu,
                            '시각적특징': row['시각적특징']
                        })
            
            self._add_activity(f"메뉴 항목 생성: {len(menu_items)}개")
            return menu_items
            
        except Exception as e:
            self.logger.error(f"메뉴 데이터 로드 실패: {e}")
            raise
    
    def generate_api_keywords(self, menu_name, api_type):
        """API별 최적화된 검색 키워드 생성 (블로그 키워드 확장)"""
        base_keywords = {
            'news': [menu_name, f"{menu_name} 맛집"],
            'blog': [
                menu_name, f"{menu_name} 후기", f"{menu_name} 먹어봤어요",
                f"{menu_name} 리뷰", f"{menu_name} 맛있어요", f"{menu_name} 추천",
                f"오늘 {menu_name}", f"{menu_name} 먹었어요"
            ],
            'image': [menu_name, f"{menu_name} 음식", f"{menu_name} 요리"]
        }
        return base_keywords.get(api_type, [menu_name])
    
    def count_period_content(self, api_type, menu_name):
        """특정 기간의 콘텐츠 수 계산"""
        unique_content_urls = set()
        search_keywords = self.generate_api_keywords(menu_name, api_type)
        
        for keyword in search_keywords:
            if self.is_stopped:
                break
                
            start = 1
            max_pages = 3
            
            for page in range(max_pages):
                if self.is_stopped or start > self.api_limits['max_start_position']:
                    break
                
                result = self.robust_api_call(api_type, keyword, start=start, display=100)
                if not result or 'items' not in result:
                    break 
                
                items = result['items']
                if not items:
                    break
                
                valid_count_in_page = 0
                
                for item in items:
                    content_url = item.get('link', '')
                    
                    if api_type == 'news':
                        date_field = item.get('pubDate', '')
                        is_valid_date = self.validate_news_date(date_field)
                    elif api_type == 'blog':
                        date_field = item.get('postdate', '')
                        is_valid_date = self.validate_blog_date(date_field)
                    else:
                        is_valid_date = False
                    
                    if (is_valid_date and content_url and content_url not in unique_content_urls):
                        unique_content_urls.add(content_url)
                        valid_count_in_page += 1
                
                if valid_count_in_page == 0:
                    break
                
                start += len(items)
                time.sleep(self.api_limits['base_delay'])
        
        return len(unique_content_urls)
    
    def analyze_menu_popularity(self, menu_item):
        """메뉴 인기도 분석"""
        menu_name = menu_item['상세메뉴']
        analysis_start = datetime.now()
        
        try:
            news_count = self.count_period_content('news', menu_name)
            if self.is_stopped:
                return None
                
            blog_count = self.count_period_content('blog', menu_name)
            if self.is_stopped:
                return None
            
            self.stats['news_content_count'] += news_count
            self.stats['blog_content_count'] += blog_count
            
            balanced_score = (
                news_count * self.api_weights['news'] +
                blog_count * self.api_weights['blog']
            )
            
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            if self.stats['successful_analyses'] > 0:
                self.stats['avg_analysis_time'] = (
                    (self.stats['avg_analysis_time'] * self.stats['successful_analyses'] + analysis_time) / 
                    (self.stats['successful_analyses'] + 1)
                )
            else:
                self.stats['avg_analysis_time'] = analysis_time
            
            self.stats['successful_analyses'] += 1
            
            return {
                'menu_name': menu_name,
                'news_count': news_count,
                'blog_count': blog_count,
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
                'balanced_score': 0.0,
                'analysis_success': False,
                'error_message': str(e)
            }
    
    def analyze_content_ratios(self):
        """실제 뉴스:블로그 콘텐츠 비율 분석"""
        if self.stats['news_content_count'] > 0 and self.stats['blog_content_count'] > 0:
            actual_ratio = self.stats['news_content_count'] / self.stats['blog_content_count']
            self.actual_content_ratios['news_to_blog_ratio'] = actual_ratio
            
            expected_ratio = self.actual_content_ratios['expected_ratio']
            ratio_difference = abs(actual_ratio - expected_ratio) / expected_ratio
            
            if ratio_difference > 0.5:
                self.actual_content_ratios['weight_adjustment_needed'] = True
                self.logger.warning(f"콘텐츠 비율 불균형 감지: 실제 {actual_ratio:.1f}:1, 기대 {expected_ratio:.1f}:1")
            
            self._add_activity(f"콘텐츠 비율: 뉴스 {self.stats['news_content_count']}, 블로그 {self.stats['blog_content_count']} (비율 {actual_ratio:.1f}:1)")
    
    def step1_analyze_popularity(self, csv_file_path):
        """1단계: 인기도 분석"""
        self._update_phase("인기도 분석 (뉴스+블로그)", 1)
        
        menu_items = self.load_menu_data(csv_file_path)
        total_menus = len(menu_items)
        
        for i, menu_item in enumerate(menu_items):
            if self.is_stopped:
                break
                
            menu_name = menu_item['상세메뉴']
            self._add_activity(f"분석: {menu_name}")
            
            popularity_result = self.analyze_menu_popularity(menu_item)
            if popularity_result is None:  # 중단된 경우
                break
                
            popularity_result.update({
                '대분류': menu_item['대분류'],
                '중분류': menu_item['중분류'],
                '소분류': menu_item['소분류'],
                '시각적특징': menu_item['시각적특징']
            })
            
            self.popularity_results.append(popularity_result)
            
            self.phase_progress = ((i + 1) / total_menus) * 100
            self.overall_progress = (self.phase_progress / self.total_phases) * (1/3)
            
            if popularity_result['analysis_success']:
                self._add_activity(f"완료: {menu_name} (점수: {popularity_result['balanced_score']:.2f})")
            else:
                self._add_activity(f"실패: {menu_name}")
            
            time.sleep(random.uniform(0.2, 0.5))
        
        if not self.is_stopped:
            self.analyze_content_ratios()
            self._add_activity(f"1단계 완료: 성공 {self.stats['successful_analyses']}, 실패 {self.stats['failed_analyses']}")
        
        return menu_items
    
    def step2_calculate_quotas(self, menu_items):
        """2단계: 할당량 계산"""
        if self.is_stopped:
            return
            
        self._update_phase("할당량 계산", 2)
        
        successful_results = [r for r in self.popularity_results if r['analysis_success']]
        total_balanced_score = sum([r['balanced_score'] for r in successful_results])
        
        self._add_activity(f"유효 분석: {len(successful_results)}개")
        self._add_activity(f"총 점수: {total_balanced_score:.2f}")
        
        base_total = int(self.target_images * self.base_allocation_ratio)
        base_quota_per_menu = max(1, base_total // len(menu_items))
        used_base_total = base_quota_per_menu * len(menu_items)
        proportional_total = self.target_images - used_base_total
        
        allocated_total = 0
        
        for result in self.popularity_results:
            menu_name = result['menu_name']
            balanced_score = result['balanced_score']
            
            base_quota = base_quota_per_menu
            
            if result['analysis_success'] and total_balanced_score > 0:
                ratio = balanced_score / total_balanced_score
                additional_quota = int(proportional_total * ratio)
            else:
                additional_quota = 0
            
            total_quota = base_quota + additional_quota
            self.menu_quotas[menu_name] = total_quota
            allocated_total += total_quota
        
        # TOP 메뉴 업데이트
        sorted_results = sorted(successful_results, key=lambda x: x['balanced_score'], reverse=True)
        self.stats['top_menus'] = [
            {
                'name': result['menu_name'],
                'score': result['balanced_score'],
                'quota': self.menu_quotas.get(result['menu_name'], 0)
            }
            for result in sorted_results[:5]
        ]
        
        self.phase_progress = 100.0
        self.overall_progress = ((1 + 1) / self.total_phases) * 100 / 3
        
        self._add_activity(f"2단계 완료: 총 {allocated_total:,}개 할당")
    
    def get_collected_count_for_menu(self, menu_name):
        """특정 메뉴의 수집된 이미지 수 반환"""
        return len([img for img in self.collected_images if img['menu_name'] == menu_name])
    
    def collect_menu_images(self, menu_name, target_quota):
        """메뉴별 이미지 수집"""
        if target_quota <= 0 or self.is_stopped:
            return []
        
        collection_start = datetime.now()
        collected_images = []
        search_keywords = self.generate_api_keywords(menu_name, 'image')
        
        for keyword in search_keywords:
            if len(collected_images) >= target_quota or self.is_stopped:
                break
            
            start = 1
            max_pages = 5
            
            for page in range(max_pages):
                if len(collected_images) >= target_quota or self.is_stopped:
                    break
                
                if start > self.api_limits['max_start_position']:
                    break
                
                result = self.robust_api_call('image', keyword, start=start, display=100)
                if not result or 'items' not in result:
                    break
                
                images = result['items']
                if not images:
                    break
                
                for img in images:
                    if len(collected_images) >= target_quota or self.is_stopped:
                        break
                    
                    img_url = img.get('link', '')
                    img_title = img.get('title', '').replace('<b>', '').replace('</b>', '')
                    
                    if (img_url and img_url not in self.collected_urls):
                        self.collected_urls.add(img_url)
                        
                        image_data = {
                            'menu_name': menu_name,
                            'image_url': img_url,
                            'title': img_title,
                            'thumbnail': img.get('thumbnail', ''),
                            'size_width': img.get('sizewidth', ''),
                            'size_height': img.get('sizeheight', ''),
                            'search_keyword': keyword,
                            'collection_type': 'retry' if self.is_retry_phase else 'normal',
                            'collected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        collected_images.append(image_data)
                
                start += len(images)
                time.sleep(self.api_limits['base_delay'])
        
        if collected_images and not self.is_stopped:
            collection_time = (datetime.now() - collection_start).total_seconds()
            if self.stats['successful_collections'] > 0:
                self.stats['avg_collection_time'] = (
                    (self.stats['avg_collection_time'] * self.stats['successful_collections'] + collection_time) / 
                    (self.stats['successful_collections'] + 1)
                )
            else:
                self.stats['avg_collection_time'] = collection_time
            
            self.stats['successful_collections'] += 1
        elif not collected_images:
            self.stats['failed_collections'] += 1
        
        return collected_images
    
    def step3_collect_images(self, menu_items):
        """3단계: 이미지 수집"""
        if self.is_stopped:
            return
            
        self._update_phase("이미지 수집 (시점 무관)", 3)
        
        sorted_quotas = sorted(self.menu_quotas.items(), key=lambda x: x[1], reverse=True)
        total_menus = len(sorted_quotas)
        
        for i, (menu_name, target_quota) in enumerate(sorted_quotas):
            if self.is_stopped:
                break
                
            if self.current_collected_count >= self.target_images:
                self._add_activity(f"목표 달성으로 완료!")
                break
            
            phase_text = "재수집" if self.is_retry_phase else "수집"
            self._add_activity(f"{phase_text}: {menu_name} (할당: {target_quota}개)")
            
            remaining_quota = self.target_images - self.current_collected_count
            actual_quota = min(target_quota, remaining_quota)
            
            try:
                images = self.collect_menu_images(menu_name, actual_quota)
                
                if images:
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
                    
                    if self.is_retry_phase:
                        self.retry_collected_count += len(images)
                        self.stats['retry_successful'] += 1
                    
                    success_rate = (len(images) / target_quota * 100) if target_quota > 0 else 0
                    self._add_activity(f"완료: {menu_name} ({len(images)}개, {success_rate:.1f}%)")
                else:
                    if self.is_retry_phase:
                        self.stats['retry_failed'] += 1
                    self._add_activity(f"실패: {menu_name}")
                
            except Exception as e:
                self.logger.error(f"메뉴 '{menu_name}' 이미지 수집 실패: {e}")
                if self.is_retry_phase:
                    self.stats['retry_failed'] += 1
                self._add_activity(f"오류: {menu_name}")
            
            self.phase_progress = ((i + 1) / total_menus) * 100
            if not self.is_retry_phase:
                self.overall_progress = ((2 + self.phase_progress/100) / self.total_phases) * 100
            
            time.sleep(random.uniform(0.2, 0.5))
        
        if not self.is_stopped:
            phase_text = "재수집" if self.is_retry_phase else "수집"
            self._add_activity(f"3단계 {phase_text} 완료: 총 {self.current_collected_count:,}개")
            
            if not self.is_retry_phase:
                self.overall_progress = 100.0
    
    def step4_retry_failed_collections(self, menu_items):
        """4단계: 0개 수집 메뉴 재수집"""
        if self.is_stopped:
            return
        
        # 0개 수집된 메뉴 찾기
        failed_menus = []
        for menu_name, quota in self.menu_quotas.items():
            collected_count = self.get_collected_count_for_menu(menu_name)
            if collected_count == 0 and quota > 0:
                failed_menus.append(menu_name)
        
        if not failed_menus:
            self._add_activity("재수집 대상 메뉴 없음")
            return
        
        self._add_activity(f"재수집 시작: {len(failed_menus)}개 메뉴")
        self.is_retry_phase = True
        self.retry_target_menus = failed_menus
        
        # 재수집용 할당량 설정
        retry_quotas = {}
        for menu_name in failed_menus:
            retry_quotas[menu_name] = self.menu_quotas[menu_name]
        
        # 기존 step3 로직 재사용
        sorted_retry_quotas = sorted(retry_quotas.items(), key=lambda x: x[1], reverse=True)
        total_retry_menus = len(sorted_retry_quotas)
        
        for i, (menu_name, target_quota) in enumerate(sorted_retry_quotas):
            if self.is_stopped:
                break
            
            self._add_activity(f"재수집: {menu_name} (할당: {target_quota}개)")
            
            try:
                images = self.collect_menu_images(menu_name, target_quota)
                
                if images:
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
                    self.retry_collected_count += len(images)
                    self.stats['retry_successful'] += 1
                    
                    success_rate = (len(images) / target_quota * 100) if target_quota > 0 else 0
                    self._add_activity(f"재수집 성공: {menu_name} ({len(images)}개, {success_rate:.1f}%)")
                else:
                    self.stats['retry_failed'] += 1
                    self._add_activity(f"재수집 실패: {menu_name}")
                
            except Exception as e:
                self.stats['retry_failed'] += 1
                self._add_activity(f"재수집 오류: {menu_name}")
            
            time.sleep(random.uniform(0.2, 0.5))
        
        if not self.is_stopped:
            self._add_activity(f"재수집 완료: {self.retry_collected_count}개 추가 수집, 성공 {self.stats['retry_successful']}, 실패 {self.stats['retry_failed']}")
        
        self.is_retry_phase = False
    
    def save_results(self, output_file=None):
        """결과를 Excel 파일로 저장"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"menu_images_{self.search_year}Q{self.search_quarter}_{timestamp}.xlsx"
        
        try:
            self._add_activity("결과 저장 중...")
            
            popularity_df = pd.DataFrame(self.popularity_results)
            popularity_df['할당량'] = popularity_df['menu_name'].map(self.menu_quotas)
            popularity_df = popularity_df.sort_values('balanced_score', ascending=False)
            
            if self.collected_images:
                images_df = pd.DataFrame(self.collected_images)
                collection_stats = images_df.groupby('menu_name').size().reset_index(name='실제수집량')
                collection_stats = collection_stats.merge(
                    popularity_df[['menu_name', '할당량', 'balanced_score']], 
                    on='menu_name', how='right'
                ).fillna(0)
                collection_stats['달성률'] = (collection_stats['실제수집량'] / collection_stats['할당량'] * 100).round(1)
                collection_stats = collection_stats.sort_values('실제수집량', ascending=False)
            else:
                images_df = pd.DataFrame()
                collection_stats = pd.DataFrame()
            
            monitoring_log = list(self.recent_activities) if self.recent_activities else []
            monitoring_df = pd.DataFrame(monitoring_log, columns=['활동로그']) if monitoring_log else pd.DataFrame()
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                popularity_df.to_excel(writer, sheet_name='인기도분석결과', index=False)
                
                if not images_df.empty:
                    images_df.to_excel(writer, sheet_name='수집된이미지', index=False)
                
                if not collection_stats.empty:
                    collection_stats.to_excel(writer, sheet_name='메뉴별수집실적', index=False)
                
                if not monitoring_df.empty:
                    monitoring_df.to_excel(writer, sheet_name='실행로그', index=False)
                
                # 전체 요약
                total_elapsed = datetime.now() - self.start_time
                summary_data = {
                    '항목': [
                        '분석 기간', '총 메뉴 수', '성공한 분석', '실패한 분석',
                        '목표 이미지', '수집 이미지', '달성률', '뉴스 콘텐츠',
                        '블로그 콘텐츠', '뉴스 가중치', '블로그 가중치', 
                        'API 요청', '재수집 대상', '재수집 성공', '재수집 실패',
                        '실행 시간', '완료 시각'
                    ],
                    '값': [
                        f"{self.search_year}년 {self.search_quarter}분기",
                        f"{len(self.popularity_results)}개",
                        f"{self.stats['successful_analyses']}개",
                        f"{self.stats['failed_analyses']}개",
                        f"{self.target_images:,}개",
                        f"{len(self.collected_images):,}개",
                        f"{(len(self.collected_images)/self.target_images*100):.1f}%",
                        f"{self.stats['news_content_count']:,}개",
                        f"{self.stats['blog_content_count']:,}개",
                        f"{self.api_weights['news']:.3f}",
                        f"{self.api_weights['blog']:.3f}",
                        f"{self.daily_request_count:,}회",
                        f"{len(self.retry_target_menus)}개",
                        f"{self.stats['retry_successful']}개",
                        f"{self.stats['retry_failed']}개",
                        f"{str(total_elapsed).split('.')[0]}",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='전체요약', index=False)
            
            self._add_activity(f"결과 저장 완료: {output_file}")
            print(f"\n결과 저장 완료: {output_file}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
    
    def run_process(self, csv_file_path):
        """전체 프로세스 실행"""
        try:
            self.is_running = True
            self._add_activity("프로세스 시작")
            
            # 1단계: 인기도 분석
            menu_items = self.step1_analyze_popularity(csv_file_path)
            if self.is_stopped:
                return
            
            # 2단계: 할당량 계산
            self.step2_calculate_quotas(menu_items)
            if self.is_stopped:
                return
            
            # 3단계: 이미지 수집
            self.step3_collect_images(menu_items)
            if self.is_stopped:
                return
            
            # 4단계: 실패 메뉴 재수집
            self.step4_retry_failed_collections(menu_items)
            if self.is_stopped:
                return
            
            # 결과 저장
            self.save_results()
            
            total_elapsed = datetime.now() - self.start_time
            self._add_activity(f"전체 프로세스 완료! 수집: {len(self.collected_images):,}개 (재수집: {self.retry_collected_count}개)")
            print(f"\n프로세스 완료! 수집: {len(self.collected_images):,}개, 시간: {str(total_elapsed).split('.')[0]}")
            
        except Exception as e:
            self.logger.error(f"프로세스 실행 중 오류: {e}")
            if self.collected_images:
                try:
                    self.save_results("partial_results.xlsx")
                    print("부분 결과 저장됨: partial_results.xlsx")
                except:
                    pass
        finally:
            self.is_running = False


def create_monitoring_window(collector):
    """회색 기반 모니터링 창 생성 (중단 기능 포함)"""
    root = tk.Tk()
    root.title("메뉴 이미지 수집 실시간 모니터링")
    root.geometry("1000x900")
    root.configure(bg='#f0f0f0')
    
    # 설정 프레임
    settings_frame = tk.Frame(root, bg='#e0e0e0', relief='ridge', bd=2)
    settings_frame.pack(fill='x', padx=10, pady=10)
    
    # 시기 설정
    tk.Label(settings_frame, text="분석 시기 설정:", bg='#e0e0e0', 
             font=('Arial', 12, 'bold'), fg='#333333').pack(anchor='w', padx=10, pady=5)
    
    control_frame = tk.Frame(settings_frame, bg='#e0e0e0')
    control_frame.pack(fill='x', padx=10, pady=5)
    
    # 연도 설정
    tk.Label(control_frame, text="연도:", bg='#e0e0e0', fg='#333333').pack(side='left', padx=(0, 5))
    year_var = tk.StringVar(value=str(collector.search_year))
    year_combo = ttk.Combobox(control_frame, textvariable=year_var, width=8, 
                             values=['2023', '2024', '2025', '2026'], state='readonly')
    year_combo.pack(side='left', padx=(0, 15))
    
    # 분기 설정
    tk.Label(control_frame, text="분기:", bg='#e0e0e0', fg='#333333').pack(side='left', padx=(0, 5))
    quarter_var = tk.StringVar(value=str(collector.search_quarter))
    quarter_combo = ttk.Combobox(control_frame, textvariable=quarter_var, width=8, 
                                values=['1', '2', '3', '4'], state='readonly')
    quarter_combo.pack(side='left', padx=(0, 15))
    
    # 목표 이미지
    tk.Label(control_frame, text="목표:", bg='#e0e0e0', fg='#333333').pack(side='left', padx=(0, 5))
    target_var = tk.StringVar(value=str(collector.target_images))
    target_combo = ttk.Combobox(control_frame, textvariable=target_var, width=8, 
                               values=['1000', '2000', '5000', '8000', '10000'], state='readonly')
    target_combo.pack(side='left', padx=(0, 15))
    
    # 버튼 프레임
    button_frame = tk.Frame(control_frame, bg='#e0e0e0')
    button_frame.pack(side='left', padx=10)
    
    # 설정 적용 함수
    def apply_settings():
        if collector.is_running:
            messagebox.showwarning("경고", "수집 진행 중에는 설정을 변경할 수 없습니다.")
            return
            
        try:
            new_year = int(year_var.get())
            new_quarter = int(quarter_var.get())
            new_target = int(target_var.get())
            
            if new_quarter not in [1, 2, 3, 4]:
                messagebox.showerror("오류", "분기는 1-4 사이여야 합니다.")
                return
            
            if new_target < 100:
                messagebox.showerror("오류", "목표는 100개 이상이어야 합니다.")
                return
            
            collector.search_year = new_year
            collector.search_quarter = new_quarter
            collector.search_months = collector.get_quarter_months(new_quarter)
            collector.target_images = new_target
            
            collector.reset_for_new_period()
            
            current_label.config(text=f"현재: {new_year}년 {new_quarter}분기, 목표: {new_target:,}개")
            start_button.config(state='normal')
            
        except ValueError:
            messagebox.showerror("오류", "올바른 숫자를 입력해주세요.")
    
    # 시작 함수
    def start_collection():
        if collector.is_running:
            messagebox.showinfo("알림", "이미 수집이 진행 중입니다.")
            return
        
        # UI 상태 변경
        year_combo.config(state='disabled')
        quarter_combo.config(state='disabled')
        target_combo.config(state='disabled')
        apply_button.config(state='disabled')
        start_button.config(state='disabled')
        stop_button.config(state='normal')
        
        # 수집 프로세스 시작
        def run_collection():
            try:
                collector.run_process(CSV_FILE_PATH)
            except Exception as e:
                print(f"수집 프로세스 오류: {e}")
            finally:
                # UI 상태 복원
                root.after(0, lambda: restore_ui_after_completion())
        
        collection_thread = threading.Thread(target=run_collection, daemon=True)
        collection_thread.start()
    
    # 중단 함수
    def stop_collection():
        collector.stop_process()
        stop_button.config(state='disabled')
    
    # 완료 후 UI 복원
    def restore_ui_after_completion():
        year_combo.config(state='readonly')
        quarter_combo.config(state='readonly')
        target_combo.config(state='readonly')
        apply_button.config(state='normal')
        start_button.config(state='normal')
        stop_button.config(state='disabled')
    
    # 버튼들
    apply_button = tk.Button(button_frame, text="설정 적용", command=apply_settings,
                            bg='#4CAF50', fg='white', font=('Arial', 10))
    apply_button.pack(side='left', padx=2)
    
    start_button = tk.Button(button_frame, text="수집 시작", command=start_collection,
                            bg='#2196F3', fg='white', font=('Arial', 10))
    start_button.pack(side='left', padx=2)
    
    stop_button = tk.Button(button_frame, text="중단", command=stop_collection,
                           bg='#f44336', fg='white', font=('Arial', 10), state='disabled')
    stop_button.pack(side='left', padx=2)
    
    # 현재 설정
    current_label = tk.Label(settings_frame, 
                           text=f"현재: {collector.search_year}년 {collector.search_quarter}분기, 목표: {collector.target_images:,}개",
                           bg='#e0e0e0', fg='#333333', font=('Arial', 10, 'bold'))
    current_label.pack(anchor='w', padx=10, pady=5)
    
    # 구분선
    tk.Frame(root, height=2, bg='#cccccc').pack(fill='x', padx=10, pady=5)
    
    # 진행률 프레임
    progress_frame = tk.Frame(root, bg='#f0f0f0')
    progress_frame.pack(fill='x', padx=10, pady=10)
    
    # 전체 진행률
    tk.Label(progress_frame, text="전체 진행률:", bg='#f0f0f0', 
             font=('Arial', 12, 'bold'), fg='#333333').pack(anchor='w')
    
    overall_progress_var = tk.DoubleVar()
    overall_progress = ttk.Progressbar(progress_frame, variable=overall_progress_var, 
                                     length=500, mode='determinate')
    overall_progress.pack(fill='x', pady=2)
    
    overall_label = tk.Label(progress_frame, text="0%", bg='#f0f0f0', fg='#333333')
    overall_label.pack(anchor='w')
    
    # 현재 단계
    phase_label = tk.Label(progress_frame, text="현재 단계: 대기 중", bg='#f0f0f0', 
                          font=('Arial', 12, 'bold'), fg='#333333')
    phase_label.pack(anchor='w', pady=(10,0))
    
    phase_progress_var = tk.DoubleVar()
    phase_progress = ttk.Progressbar(progress_frame, variable=phase_progress_var, 
                                   length=500, mode='determinate')
    phase_progress.pack(fill='x', pady=2)
    
    phase_progress_label = tk.Label(progress_frame, text="0%", bg='#f0f0f0', fg='#333333')
    phase_progress_label.pack(anchor='w')
    
    # 통계 프레임
    stats_frame = tk.Frame(root, bg='#f0f0f0')
    stats_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    tk.Label(stats_frame, text="실시간 통계:", bg='#f0f0f0', 
             font=('Arial', 12, 'bold'), fg='#333333').pack(anchor='w')
    
    stats_text = scrolledtext.ScrolledText(stats_frame, height=12, width=90, 
                                         bg='#ffffff', fg='#333333', 
                                         font=('Consolas', 10))
    stats_text.pack(fill='both', expand=True, pady=2)
    
    # 로그 프레임
    log_frame = tk.Frame(root, bg='#f0f0f0')
    log_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    tk.Label(log_frame, text="실시간 로그:", bg='#f0f0f0', 
             font=('Arial', 12, 'bold'), fg='#333333').pack(anchor='w')
    
    log_text = scrolledtext.ScrolledText(log_frame, height=8, width=90, 
                                       bg='#ffffff', fg='#333333', 
                                       font=('Consolas', 9))
    log_text.pack(fill='both', expand=True, pady=2)
    
    def update_display():
        """실시간 디스플레이 업데이트"""
        try:
            # 진행률 업데이트
            overall_progress_var.set(collector.overall_progress)
            overall_label.config(text=f"{collector.overall_progress:.1f}%")
            
            phase_progress_var.set(collector.phase_progress)
            phase_progress_label.config(text=f"{collector.phase_progress:.1f}%")
            
            phase_text = collector.current_phase
            if collector.is_retry_phase:
                phase_text += " (재수집)"
            phase_label.config(text=f"현재 단계: {phase_text} ({collector.current_phase_num}/{collector.total_phases})")
            
            # 통계 업데이트
            elapsed_time = datetime.now() - collector.start_time
            
            stats_content = f"""수집 현황:
  이미지: {collector.current_collected_count:,}/{collector.target_images:,}개 ({collector.current_collected_count/collector.target_images*100:.1f}%)
  재수집으로 추가: {collector.retry_collected_count}개
  분석된 메뉴: {len(collector.popularity_results)}개
  API 요청: {collector.daily_request_count:,}회
  경과 시간: {str(elapsed_time).split('.')[0]}

분석 설정:
  분석 시기: {collector.search_year}년 {collector.search_quarter}분기
  목표 이미지: {collector.target_images:,}개

분석 통계:
  뉴스 콘텐츠: {collector.stats['news_content_count']:,}개
  블로그 콘텐츠: {collector.stats['blog_content_count']:,}개
  성공한 분석: {collector.stats['successful_analyses']}개
  실패한 분석: {collector.stats['failed_analyses']}개

재수집 현황:
  재수집 대상: {len(collector.retry_target_menus)}개 메뉴
  재수집 성공: {collector.stats['retry_successful']}개
  재수집 실패: {collector.stats['retry_failed']}개

가중치 정보:
  뉴스 가중치: {collector.api_weights['news']:.3f}
  블로그 가중치: {collector.api_weights['blog']:.3f}

TOP 5 인기 메뉴:"""
            
            for i, menu_info in enumerate(collector.stats['top_menus'][:5]):
                collected = collector.get_collected_count_for_menu(menu_info['name'])
                stats_content += f"\n  {i+1}. {menu_info['name']}: {menu_info['score']:.2f}점 (할당: {menu_info['quota']}개, 수집: {collected}개)"
            
            if len(collector.stats['top_menus']) == 0:
                stats_content += "\n  분석 중..."
            
            stats_text.delete(1.0, tk.END)
            stats_text.insert(tk.END, stats_content)
            
            # 로그 업데이트
            log_text.delete(1.0, tk.END)
            for activity in list(collector.recent_activities)[-15:]:
                log_text.insert(tk.END, activity + '\n')
            log_text.see(tk.END)
            
            root.after(1000, update_display)
            
        except Exception as e:
            print(f"모니터링 업데이트 오류: {e}")
            root.after(1000, update_display)
    
    def on_closing():
        if collector.is_running:
            if messagebox.askokcancel("종료", "수집이 진행 중입니다. 정말 종료하시겠습니까?"):
                collector.stop_process()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    update_display()
    
    return root


# 전역 변수
CSV_FILE_PATH = "식당대12중53소132상세메뉴379분류.csv"


def main():
    """메인 실행 함수"""
    try:
        print("="*70)
        print("              메뉴 이미지 수집 시스템 - 최종 완성 버전")
        print("="*70)
        print("주요 특징:")
        print("- 디폴트: 2025년 1분기")
        print("- 0개 수집 메뉴 자동 재수집")
        print("- 연속 분기 작업 지원")
        print("- 중단 기능 포함")
        print("- 회색 기반 UI")
        print("="*70)
        
        # 파일 확인
        if not os.path.exists(CSV_FILE_PATH):
            print(f"오류: CSV 파일을 찾을 수 없습니다 - {CSV_FILE_PATH}")
            return
        
        if not os.path.exists('.env'):
            print("오류: .env 파일이 필요합니다.")
            print("Client_ID=your_client_id")
            print("Client_Secret=your_client_secret")
            return
        
        # 시스템 초기화
        collector = MenuImageCollectorFinal()
        
        # 모니터링 창 생성
        monitor_window = create_monitoring_window(collector)
        
        print("모니터링 창이 열립니다...")
        print("시기 설정 후 '수집 시작' 버튼을 클릭하세요.")
        monitor_window.mainloop()
        
    except Exception as e:
        print(f"실행 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
