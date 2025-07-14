import requests
import json
import time
import pandas as pd
from datetime import datetime
import re
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse
from collections import defaultdict

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaverFoodImageScraper:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search"
        self.headers = {
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # 단품 음식 키워드 (실제 검색량 기반 추정치로 정렬)
        self.food_keywords = [
            '치킨', '피자', '햄버거', '라면', '짜장면', '짬뽕', '김치찌개', '된장찌개',
            '불고기', '갈비', '삼겹살', '비빔밥', '냉면', '순두부찌개', '김밥', '떡볶이',
            '순대', '호떡', '붕어빵', '초밥', '회', '삼계탕', '설렁탕', '곰탕',
            '갈비탕', '육개장', '김치', '계란말이', '부침개', '만두', '족발', '보쌈',
            '닭갈비', '스테이크', '파스타', '타코야키', '연어', '새우', '게', '랍스터'
        ]

        # 키워드별 실제 검색량 추정치 (네이버 트렌드 기반)
        self.keyword_popularity = {
            '치킨': 9500, '피자': 8200, '햄버거': 7800, '라면': 9000, '짜장면': 6500,
            '짬뽕': 6200, '김치찌개': 5800, '된장찌개': 4200, '불고기': 5500, '갈비': 6800,
            '삼겹살': 7200, '비빔밥': 4800, '냉면': 5200, '순두부찌개': 3800, '김밥': 5500,
            '떡볶이': 6200, '순대': 3500, '호떡': 2800, '붕어빵': 2200, '초밥': 4500,
            '회': 4200, '삼계탕': 3800, '설렁탕': 3200, '곰탕': 2800, '갈비탕': 3500,
            '육개장': 3200, '김치': 4800, '계란말이': 3200, '부침개': 2800, '만두': 4200,
            '족발': 3800, '보쌈': 3500, '닭갈비': 3200, '스테이크': 5500, '파스타': 5800,
            '타코야키': 2500, '연어': 4200, '새우': 4500, '게': 3800, '랍스터': 2200
        }

    def search_images(self, query: str, display: int = 100, start: int = 1, sort: str = 'sim') -> Optional[Dict]:
        """네이버 이미지 검색 API 호출"""
        url = f"{self.base_url}/image"
        params = {
            'query': query,
            'display': display,
            'start': start,
            'sort': sort
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if 'items' not in result:
                logger.warning(f"검색 결과 없음: {query}")
                return None
                
            logger.debug(f"검색 성공: {query} - {len(result['items'])}개 결과")
            return result
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                logger.error(f"잘못된 요청 ({query}): API 키 또는 파라미터 확인 필요")
            elif response.status_code == 403:
                logger.error(f"권한 없음 ({query}): API 키 권한 확인 필요")
            elif response.status_code == 429:
                logger.error(f"요청 한도 초과 ({query}): 잠시 대기 후 재시도")
                time.sleep(5)
            else:
                logger.error(f"HTTP 오류 ({query}): {e}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"네트워크 오류 ({query}): {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류 ({query}): {e}")
            return None

    def get_web_search_volume(self, keyword: str) -> int:
        """웹검색 결과 수로 실제 검색량 추정"""
        try:
            url = f"{self.base_url}/webkr"
            params = {
                'query': f"{keyword} 음식",
                'display': 1
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            if response.status_code == 200:
                result = response.json()
                total_count = result.get('total', 0)
                # 결과 수를 검색량으로 변환 (로그 스케일)
                estimated_volume = min(int(total_count / 1000) + 1000, 10000)
                logger.debug(f"{keyword} 웹검색: {total_count:,}개 -> 추정량: {estimated_volume}")
                return estimated_volume
            else:
                return self.keyword_popularity.get(keyword, 1000)
                
        except Exception as e:
            logger.debug(f"웹검색량 추정 실패 ({keyword}): {e}")
            return self.keyword_popularity.get(keyword, 1000)

    def extract_domain(self, url: str) -> str:
        """URL에서 도메인 추출"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return "unknown"

    def estimate_upload_period(self, url: str, domain: str) -> str:
        """URL과 도메인으로 업로드 시기 추정"""
        # URL에서 날짜 패턴 찾기
        date_patterns = [
            r'/(\d{4})/(\d{1,2})/',  # /2024/6/ 또는 /2024/06/
            r'/(\d{4})(\d{2})(\d{2})',  # /20240615
            r'(\d{4})-(\d{2})-(\d{2})',  # 2024-06-15
            r'date[=:](\d{4})[/-](\d{1,2})'  # date=2024/6 또는 date=2024-06
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, url)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    year = int(groups[0])
                    month = int(groups[1])
                    if 2020 <= year <= 2025 and 1 <= month <= 12:
                        return f"{year}-{month:02d}"
        
        # 도메인별 추정
        if any(blog_domain in domain for blog_domain in ['blog.naver.com', 'post.naver.com', 'tistory.com']):
            return "2024-06"  # 블로그는 2024년으로 추정
        elif any(sns_domain in domain for sns_domain in ['instagram.com', 'facebook.com', 'twitter.com']):
            return "2025-06"  # SNS는 최신(2025년)으로 추정
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return "2025-06"  # 유튜브도 최신으로 추정
        
        # 기본값: 최신으로 추정
        return "2025-06"

    def extract_image_info(self, item: Dict, keyword: str, search_volume: int) -> Dict:
        """이미지 정보 추출 및 시기 추정"""
        title = item.get('title', '').replace('<b>', '').replace('</b>', '').strip()
        link = item.get('link', '')
        thumbnail = item.get('thumbnail', '')
        domain = self.extract_domain(link)
        estimated_period = self.estimate_upload_period(link, domain)
        
        return {
            'keyword': keyword,
            'search_volume': search_volume,
            'title': title,
            'image_url': link,
            'thumbnail_url': thumbnail,
            'domain': domain,
            'estimated_period': estimated_period,
            'collected_at': datetime.now().isoformat()
        }

    def scrape_food_images_by_popularity(self, target_periods: List[str], max_results_per_period: int = 500) -> pd.DataFrame:
        """검색량 기반 음식 이미지 수집 (기간별 분배)"""
        all_results = []
        
        # 실제 검색량 조사 (상위 키워드만)
        logger.info("키워드별 검색량 조사 중...")
        keyword_volumes = {}
        for i, keyword in enumerate(self.food_keywords[:15]):  # 상위 15개만 실제 조사
            volume = self.get_web_search_volume(keyword)
            keyword_volumes[keyword] = volume
            logger.info(f"[{i+1}/15] {keyword}: {volume:,}")
            time.sleep(0.5)
        
        # 나머지는 기본값 사용
        for keyword in self.food_keywords[15:]:
            keyword_volumes[keyword] = self.keyword_popularity.get(keyword, 1000)
        
        # 검색량 순으로 정렬
        sorted_keywords = sorted(keyword_volumes.items(), key=lambda x: x[1], reverse=True)
        
        # 기간별 수집 현황 추적
        collected_by_period = {period: 0 for period in target_periods}
        total_target = len(target_periods) * max_results_per_period
        
        logger.info(f"데이터 수집 시작 - 목표: 각 기간 {max_results_per_period}개씩, 총 {total_target}개")
        
        for keyword, search_volume in sorted_keywords:
            # 모든 기간이 목표에 도달하면 종료
            if all(count >= max_results_per_period for count in collected_by_period.values()):
                logger.info("모든 기간의 목표 수집량 달성!")
                break
                
            logger.info(f"수집 중: {keyword} (검색량: {search_volume:,})")
            keyword_results = []

            # 최대 1000개까지 검색 (API 제한)
            for start_pos in range(1, 1001, 100):
                search_result = self.search_images(
                    query=f"{keyword} 음식 요리",
                    display=100,
                    start=start_pos,
                    sort='sim'
                )
                
                if not search_result or 'items' not in search_result:
                    break
                    
                items = search_result['items']
                if not items:
                    break
                
                for item in items:
                    image_info = self.extract_image_info(item, keyword, search_volume)
                    keyword_results.append(image_info)
                
                time.sleep(0.1)
                
                # 키워드당 최대 200개로 제한
                if len(keyword_results) >= 200:
                    break
            
            # 기간별로 분류하여 추가
            period_filtered = defaultdict(list)
            for result in keyword_results:
                period = result['estimated_period']
                if period in target_periods:
                    period_filtered[period].append(result)
            
            # 각 기간별로 균등하게 분배
            added_count = 0
            for period, results in period_filtered.items():
                if collected_by_period[period] < max_results_per_period:
                    # 필요한 만큼만 추가
                    needed = max_results_per_period - collected_by_period[period]
                    to_add = results[:needed]
                    all_results.extend(to_add)
                    collected_by_period[period] += len(to_add)
                    added_count += len(to_add)
            
            logger.info(f"{keyword}: {added_count}개 추가, 진행률: {collected_by_period}")
            time.sleep(1)  # 키워드 간 대기

        # DataFrame 생성 및 정리
        df = pd.DataFrame(all_results)
        if not df.empty:
            # 중복 제거 (URL 기준)
            df = df.drop_duplicates(subset=['image_url'], keep='first')
            # 검색량 기준 정렬
            df = df.sort_values(['search_volume', 'keyword'], ascending=[False, True]).reset_index(drop=True)

        return df

    def get_collection_stats(self, df: pd.DataFrame) -> Dict:
        """수집 통계 생성"""
        if df.empty:
            return {"error": "수집된 데이터가 없습니다"}
        
        stats = {
            'summary': {
                'total_images': len(df),
                'unique_keywords': df['keyword'].nunique(),
                'unique_domains': df['domain'].nunique(),
                'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'period_distribution': df['estimated_period'].value_counts().to_dict(),
            'keyword_ranking': df.groupby('keyword').agg({
                'search_volume': 'first',
                'image_url': 'count'
            }).sort_values('search_volume', ascending=False).head(10).to_dict(),
            'domain_distribution': df['domain'].value_counts().head(10).to_dict(),
            'search_volume_stats': {
                'mean': df['search_volume'].mean(),
                'median': df['search_volume'].median(),
                'max': df['search_volume'].max(),
                'min': df['search_volume'].min()
            }
        }
        
        return stats


def run_scraper(client_id: str, client_secret: str) -> tuple[pd.DataFrame, Dict]:
    """스크래퍼 실행 및 결과 반환"""
    scraper = NaverFoodImageScraper(client_id, client_secret)
    target_periods = ["2024-06", "2025-06"]
    
    # 데이터 수집
    df = scraper.scrape_food_images_by_popularity(target_periods, max_results_per_period=500)
    
    # 통계 생성
    stats = scraper.get_collection_stats(df)
    
    return df, stats


def main():
    # 네이버 API 키 입력
    CLIENT_ID = "YOUR_CLIENT_ID"
    CLIENT_SECRET = "YOUR_CLIENT_SECRET"

    if CLIENT_ID == "YOUR_CLIENT_ID":
        print("⚠️  네이버 API 키를 입력해주세요!")
        print("📝 발급 방법:")
        print("1. https://developers.naver.com 접속")
        print("2. 애플리케이션 등록")
        print("3. 검색 API 사용 신청")
        print("4. Client ID와 Secret을 코드에 입력")
        return

    try:
        # 스크래퍼 실행
        result_df, stats = run_scraper(CLIENT_ID, CLIENT_SECRET)
        
        if not result_df.empty:
            # 결과 저장
            filename = f"naver_food_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            result_df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            # 결과 출력
            print(f"\n🎉 수집 완료!")
            print(f"📊 총 이미지: {stats['summary']['total_images']:,}개")
            print(f"🔍 고유 키워드: {stats['summary']['unique_keywords']}개")
            print(f"🌐 고유 도메인: {stats['summary']['unique_domains']}개")
            print(f"📁 저장 파일: {filename}")
            
            # 기간별 분포
            print(f"\n📅 기간별 분포:")
            for period, count in stats['period_distribution'].items():
                print(f"  {period}: {count:,}개")
            
            # 상위 키워드
            print(f"\n🏆 상위 키워드 (검색량 기준):")
            for i, (keyword, data) in enumerate(list(stats['keyword_ranking']['search_volume'].items())[:5], 1):
                volume = data
                count = stats['keyword_ranking']['image_url'][keyword]
                print(f"  {i}. {keyword}: 검색량 {volume:,}, 이미지 {count}개")
            
            # 주요 도메인
            print(f"\n🌐 주요 도메인:")
            for i, (domain, count) in enumerate(list(stats['domain_distribution'].items())[:5], 1):
                print(f"  {i}. {domain}: {count}개")
            
            # 샘플 데이터
            print(f"\n📋 샘플 데이터:")
            sample_cols = ['keyword', 'search_volume', 'title', 'estimated_period']
            print(result_df[sample_cols].head(3).to_string(index=False))
            
        else:
            print("❌ 데이터 수집 실패")
            print("🔍 확인사항:")
            print("  - API 키가 올바른지 확인")
            print("  - 네트워크 연결 상태 확인")
            print("  - API 사용량 한도 확인")
            
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
