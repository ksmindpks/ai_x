import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import csv
import re
import time
import threading
from urllib.parse import quote, urlparse
from datetime import datetime
from calendar import monthrange
import os
from io import BytesIO
import requests
from PIL import Image, ImageTk

# 패키지 확인
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

try:
    import pandas as pd
    EXCEL_OK = True
except ImportError:
    EXCEL_OK = False

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

class SimpleCollector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("음식 이미지 수집기")
        self.root.geometry("900x700")
        
        self.driver = None
        self.running = False
        self.menus = []
        
        self.setup_gui()
        self.load_csv()
    
    def setup_gui(self):
        """GUI 구성"""
        # 메인 프레임
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 좌측 프레임 (설정 + 로그)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 우측 프레임 (이미지 표시)
        right_frame = tk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # === 좌측: 설정 및 로그 ===
        # 제목
        tk.Label(left_frame, text="음식 이미지 수집기", font=("Arial", 14, "bold")).pack(pady=10)
        
        # CSV 상태
        self.csv_label = tk.Label(left_frame, text="CSV 확인 중...")
        self.csv_label.pack()
        
        # 기간 설정 프레임
        frame = tk.Frame(left_frame)
        frame.pack(pady=20)
        
        # 시작 기간
        tk.Label(frame, text="시작:").grid(row=0, column=0, padx=5)
        self.start_year = tk.Entry(frame, width=6)
        self.start_year.grid(row=0, column=1, padx=2)
        self.start_year.insert(0, "2024")
        tk.Label(frame, text="년").grid(row=0, column=2)
        
        self.start_month = tk.Entry(frame, width=4)
        self.start_month.grid(row=0, column=3, padx=2)
        self.start_month.insert(0, "6")
        tk.Label(frame, text="월").grid(row=0, column=4)
        
        # 끝 기간
        tk.Label(frame, text="끝:").grid(row=1, column=0, padx=5, pady=5)
        self.end_year = tk.Entry(frame, width=6)
        self.end_year.grid(row=1, column=1, padx=2, pady=5)
        self.end_year.insert(0, "2024")
        tk.Label(frame, text="년").grid(row=1, column=2, pady=5)
        
        self.end_month = tk.Entry(frame, width=4)
        self.end_month.grid(row=1, column=3, padx=2, pady=5)
        self.end_month.insert(0, "6")
        tk.Label(frame, text="월").grid(row=1, column=4, pady=5)
        
        # 버튼
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="수집 시작", command=self.start_collection, 
                                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="중지", command=self.stop_collection, 
                                 state=tk.DISABLED, bg="#f44336", fg="white")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 상태
        self.status_label = tk.Label(left_frame, text="대기 중", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # 진행률
        self.progress = ttk.Progressbar(left_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # 로그
        tk.Label(left_frame, text="진행 상황", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.log_text = scrolledtext.ScrolledText(left_frame, height=15, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # === 우측: 이미지 표시 ===
        # 현재 메뉴
        tk.Label(right_frame, text="현재 처리 메뉴", font=("Arial", 12, "bold")).pack(pady=5)
        self.current_menu_label = tk.Label(right_frame, text="대기 중", font=("Arial", 10))
        self.current_menu_label.pack()
        
        # 첫 번째 이미지
        tk.Label(right_frame, text="첫 번째 이미지", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        self.first_image_label = tk.Label(right_frame, text="이미지 없음", 
                                         width=25, height=8, bg="lightgray", relief="solid")
        self.first_image_label.pack(pady=5)
        
        # 마지막 이미지
        tk.Label(right_frame, text="마지막 이미지", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        self.last_image_label = tk.Label(right_frame, text="이미지 없음", 
                                        width=25, height=8, bg="lightgray", relief="solid")
        self.last_image_label.pack(pady=5)
        
        # 통계 정보
        tk.Label(right_frame, text="수집 통계", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        self.stats_label = tk.Label(right_frame, text="검색결과: 0개\n수집URL: 0개\n수집률: 0%", 
                                   justify=tk.LEFT, font=("Arial", 9))
        self.stats_label.pack()
    
    def load_csv(self):
        """CSV 파일 로드"""
        try:
            with open('식당대12중53소132상세메뉴379분류.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                menus = []
                seen = set()
                
                for row in reader:
                    if '상세메뉴' in row and row['상세메뉴']:
                        items = [item.strip() for item in row['상세메뉴'].split(',') if item.strip()]
                        for item in items:
                            if item not in seen and len(item) >= 2:
                                seen.add(item)
                                menus.append(item)
                
                self.menus = menus
                self.csv_label.config(text=f"메뉴 {len(menus)}개 로드 완료", fg="green")
                self.log(f"CSV 로드 완료: {len(menus)}개 메뉴")
        except Exception as e:
            self.csv_label.config(text=f"CSV 로드 실패: {e}", fg="red")
            self.log(f"CSV 로드 실패: {e}")
    
    def log(self, msg, level="INFO"):
        """상세 로그 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 레벨에 따른 텍스트 접두사
        if level == "ERROR":
            prefix = "[ERROR]"
        elif level == "SUCCESS":
            prefix = "[OK]"
        elif level == "PROGRESS":
            prefix = "[WORK]"
        else:
            prefix = "[INFO]"
        
        log_msg = f"[{timestamp}] {prefix} {msg}\n"
        
        self.log_text.insert(tk.END, log_msg)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def load_and_display_image(self, url, label, size=(150, 100)):
        """이미지 로드 및 표시"""
        if not PIL_OK:
            label.config(text="PIL 패키지 필요", bg="lightcoral")
            return
        
        try:
            # 이미지 다운로드 (타임아웃 짧게)
            response = requests.get(url, timeout=3, stream=True)
            response.raise_for_status()
            
            # PIL로 이미지 처리
            image = Image.open(BytesIO(response.content))
            image = image.convert('RGB')  # RGBA -> RGB 변환
            
            # 크기 조정 (비율 유지)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # tkinter용 이미지로 변환
            photo = ImageTk.PhotoImage(image)
            
            # 라벨에 표시
            label.config(image=photo, text="")
            label.image = photo  # 참조 유지
            
        except Exception as e:
            label.config(text=f"로드 실패\n{str(e)[:20]}...", bg="lightcoral")
    
    def update_menu_display(self, menu, first_url=None, last_url=None, stats=None):
        """메뉴 표시 업데이트"""
        # 현재 메뉴
        self.current_menu_label.config(text=menu)
        
        # 이미지 초기화
        self.first_image_label.config(image="", text="이미지 없음", bg="lightgray")
        self.last_image_label.config(image="", text="이미지 없음", bg="lightgray")
        
        # 첫 번째 이미지
        if first_url:
            self.first_image_label.config(text="로딩 중...", bg="lightyellow")
            threading.Thread(target=self.load_and_display_image, 
                           args=(first_url, self.first_image_label), daemon=True).start()
        
        # 마지막 이미지
        if last_url and last_url != first_url:
            self.last_image_label.config(text="로딩 중...", bg="lightyellow")
            threading.Thread(target=self.load_and_display_image, 
                           args=(last_url, self.last_image_label), daemon=True).start()
        elif last_url == first_url:
            self.last_image_label.config(text="첫 번째와 동일", bg="lightblue")
        
        # 통계 업데이트
        if stats:
            stats_text = f"검색결과: {stats['count']:,}개\n"
            stats_text += f"수집URL: {stats['urls']}개\n"
            stats_text += f"수집률: {stats['rate']:.1f}%"
            self.stats_label.config(text=stats_text)
    
    def validate_input(self):
        """입력 검증"""
        try:
            start_year = int(self.start_year.get())
            start_month = int(self.start_month.get())
            end_year = int(self.end_year.get())
            end_month = int(self.end_month.get())
            
            if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
                messagebox.showerror("오류", "월은 1-12 사이여야 합니다.")
                return None
            
            if datetime(start_year, start_month, 1) > datetime(end_year, end_month, 1):
                messagebox.showerror("오류", "시작 날짜가 끝 날짜보다 늦습니다.")
                return None
            
            return start_year, start_month, end_year, end_month
        except ValueError:
            messagebox.showerror("오류", "숫자를 입력하세요.")
            return None
    
    def generate_months(self, start_year, start_month, end_year, end_month):
        """월별 범위 생성"""
        months = []
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)
        
        while current <= end:
            year, month = current.year, current.month
            last_day = monthrange(year, month)[1]
            
            months.append({
                'display': f"{year}-{month:02d}",
                'start_date': f"{year}{month:02d}01",
                'end_date': f"{year}{month:02d}{last_day:02d}"
            })
            
            if month == 12:
                current = datetime(year + 1, 1, 1)
            else:
                current = datetime(year, month + 1, 1)
        
        return months
    
    def setup_driver(self):
        """드라이버 설정"""
        if not SELENIUM_OK:
            self.log("selenium 패키지가 필요합니다", "ERROR")
            return False
        
        try:
            self.log("Chrome 드라이버 설정 중...", "PROGRESS")
            
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-logging')
            options.add_argument('--log-level=3')
            
            self.driver = webdriver.Chrome(options=options)
            self.log("Chrome 드라이버 준비 완료", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"드라이버 설정 실패: {e}", "ERROR")
            return False
    
    def get_actual_image_count(self):
        """실제 이미지 개수 직접 카운트"""
        try:
            # 현재 페이지의 고유한 이미지 URL들을 수집
            unique_urls = set()
            last_count = 0
            stable_count = 0
            scroll_count = 0
            max_scrolls = 50  # 최대 스크롤 횟수
            
            self.log("   실제 이미지 개수 카운트 시작...")
            
            while scroll_count < max_scrolls:
                if not self.running:
                    break
                
                # 현재 화면의 이미지 요소들 찾기
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, 
                        'img[src*="http"], img[data-src*="http"], [data-source*="http"]')
                    
                    for elem in elements:
                        for attr in ['src', 'data-src', 'data-source']:
                            url = elem.get_attribute(attr)
                            if url and self.is_valid_url(url):
                                unique_urls.add(url)
                
                except Exception as e:
                    self.log(f"   요소 찾기 오류: {e}")
                    break
                
                current_count = len(unique_urls)
                
                # 진행 상황 로그 (10번마다)
                if scroll_count % 10 == 0:
                    self.log(f"   스크롤 {scroll_count+1}/{max_scrolls} - 현재 {current_count}개 발견")
                
                # 더 이상 새로운 이미지가 없으면 종료
                if current_count == last_count:
                    stable_count += 1
                    if stable_count >= 5:  # 5번 연속 변화 없으면 종료
                        self.log(f"   더 이상 새로운 이미지가 없어 스크롤 종료")
                        break
                else:
                    stable_count = 0
                    last_count = current_count
                
                # 스크롤 다운
                try:
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1.5)  # 로딩 대기
                    
                    # 더보기 버튼이 있으면 클릭
                    self._try_click_more_button()
                    
                except Exception as e:
                    self.log(f"   스크롤 오류: {e}")
                    break
                
                scroll_count += 1
            
            final_count = len(unique_urls)
            self.log(f"   실제 카운트 완료: {final_count}개 이미지 확인됨")
            
            return final_count, list(unique_urls)
            
        except Exception as e:
            self.log(f"   실제 카운트 실패: {e}", "ERROR")
            return 0, []
    
    def _try_click_more_button(self):
        """더보기 버튼 클릭 시도"""
        try:
            # 다양한 더보기 버튼 선택자
            more_selectors = [
                'button[class*="more"]',
                'a[class*="more"]', 
                '.btn_more',
                'button:contains("더보기")',
                'a:contains("더보기")',
                '[class*="paging"] button',
                '[class*="next"] button'
            ]
            
            for selector in more_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            self.driver.execute_script("arguments[0].click();", element)
                            time.sleep(1)
                            return True
                except:
                    continue
            
            return False
        except:
            return False
    
    def collect_urls(self):
        """URL 수집 (더 이상 사용하지 않음 - get_actual_image_count로 대체)"""
        # 이 함수는 get_actual_image_count로 대체되었습니다
        return []
    
    def is_valid_url(self, url):
        """URL 유효성 확인"""
        if len(url) < 20:
            return False
        
        url_lower = url.lower()
        
        # 제외 키워드
        if any(word in url_lower for word in ['icon', 'logo', 'banner']):
            return False
        
        # 포함 키워드
        return any(word in url_lower for word in ['blogfiles', 'postfiles', 'pstatic', 'jpg', 'png'])
    
    def collect_menu(self, menu, start_date, end_date):
        """메뉴별 수집 (실제 카운트 방식으로 변경)"""
        if not self.driver or not self.running:
            return {'menu': menu, 'estimated_count': 0, 'actual_count': 0, 'urls': []}
        
        try:
            self.log(f"'{menu}' 검색 시작", "PROGRESS")
            
            query = f"음식 {menu}"
            nso = f"so:r,p:from{start_date}to{end_date}"
            url = f"https://search.naver.com/search.naver?where=image&query={quote(query)}&sm=tab_opt&nso={quote(nso)}"
            
            self.log(f"   검색어: {query}")
            
            # 페이지 로드
            self.driver.get(url)
            time.sleep(3)
            
            # 네이버 추정 개수 (참고용)
            estimated_count = self._get_estimated_count()
            self.log(f"   네이버 추정: {estimated_count:,}개 (참고용)")
            
            # 실제 이미지 카운트 및 URL 수집
            actual_count, urls = self.get_actual_image_count()
            
            if actual_count == 0:
                self.log(f"'{menu}' 실제 이미지 없음")
                self.update_menu_display(menu)
                return {'menu': menu, 'estimated_count': estimated_count, 'actual_count': 0, 'urls': []}
            
            result = {
                'menu': menu,
                'estimated_count': estimated_count,
                'actual_count': actual_count,
                'urls': urls
            }
            
            # 결과 로그
            accuracy = (actual_count / estimated_count * 100) if estimated_count > 0 else 0
            self.log(f"'{menu}' 완료 - 실제 {actual_count}개 확인 (추정 대비 {accuracy:.1f}%)", "SUCCESS")
            
            # GUI 업데이트
            first_url = urls[0] if urls else None
            last_url = urls[-1] if urls else None
            stats = {
                'count': actual_count,
                'urls': len(urls),
                'rate': 100.0  # 실제 카운트이므로 100%
            }
            
            self.update_menu_display(menu, first_url, last_url, stats)
            
            return result
            
        except Exception as e:
            self.log(f"'{menu}' 오류: {e}", "ERROR")
            return {'menu': menu, 'estimated_count': 0, 'actual_count': 0, 'urls': [], 'error': str(e)}
    
    def _get_estimated_count(self):
        """네이버 추정 개수 가져오기 (참고용)"""
        try:
            text = self.driver.find_element(By.TAG_NAME, 'body').text
            match = re.search(r'약\s*([\d,]+)\s*개', text)
            return int(match.group(1).replace(',', '')) if match else 0
        except:
            return 0
    
    def save_excel(self, results, month_display):
        """엑셀 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"menu_images_{month_display.replace('-', '')}_{timestamp}.xlsx"
        
        try:
            if EXCEL_OK:
                self.log(f"엑셀 파일 저장 중: {filename}", "PROGRESS")
                
                # 요약 데이터
                summary = []
                url_list = []
                
                for result in results:
                    menu = result['menu']
                    estimated_count = result.get('estimated_count', 0)
                    actual_count = result.get('actual_count', 0)
                    urls = result['urls']
                    
                    summary.append({
                        '메뉴': menu,
                        '네이버추정': estimated_count,
                        '실제확인': actual_count,
                        'URL수': len(urls),
                        '정확도': f"{(actual_count/estimated_count*100):.1f}%" if estimated_count > 0 else "N/A"
                    })
                    
                    for i, url in enumerate(urls, 1):
                        url_list.append({
                            '메뉴': menu,
                            '번호': i,
                            'URL': url,
                            '도메인': urlparse(url).netloc
                        })
                
                # 엑셀 저장
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    pd.DataFrame(summary).to_excel(writer, sheet_name='요약', index=False)
                    if url_list:
                        pd.DataFrame(url_list).to_excel(writer, sheet_name='URL목록', index=False)
                
                self.log(f"엑셀 파일 저장 완료: {filename}", "SUCCESS")
                return filename
            else:
                self.log("pandas 패키지가 없어 엑셀 저장 불가", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"엑셀 저장 실패: {e}", "ERROR")
            return None
    
    def start_collection(self):
        """수집 시작"""
        if not self.menus:
            messagebox.showerror("오류", "메뉴가 없습니다")
            return
        
        # 입력 검증
        dates = self.validate_input()
        if not dates:
            return
        
        # 월별 범위
        months = self.generate_months(*dates)
        
        # 확인
        msg = f"{len(self.menus)}개 메뉴 × {len(months)}개월을 수집하시겠습니까?\n"
        msg += f"예상 소요시간: {len(self.menus) * len(months) * 3 // 60}분"
        if not messagebox.askyesno("확인", msg):
            return
        
        # 상태 변경
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        
        # 로그 시작
        self.log("=" * 50)
        self.log("음식 이미지 수집 시작", "SUCCESS")
        self.log(f"대상 기간: {dates[0]}-{dates[1]:02d} ~ {dates[2]}-{dates[3]:02d}")
        self.log(f"처리 메뉴: {len(self.menus)}개")
        self.log(f"처리 월수: {len(months)}개월")
        self.log("=" * 50)
        
        # 스레드 시작
        thread = threading.Thread(target=self.run_collection, args=(months,))
        thread.daemon = True
        thread.start()
    
    def run_collection(self, months):
        """수집 실행"""
        try:
            if not self.setup_driver():
                return
            
            total_tasks = len(months) * len(self.menus)
            current_task = 0
            self.progress['maximum'] = total_tasks
            
            saved_files = []
            
            for month_idx, month in enumerate(months, 1):
                if not self.running:
                    break
                
                self.log(f"\n[MONTH] {month['display']} 처리 시작 ({month_idx}/{len(months)})")
                self.status_label.config(text=f"{month['display']} 처리 중")
                
                results = []
                month_start_time = time.time()
                
                for menu_idx, menu in enumerate(self.menus, 1):
                    if not self.running:
                        break
                    
                    current_task += 1
                    self.progress['value'] = current_task
                    
                    # 상태 업데이트
                    self.status_label.config(text=f"{month['display']} - {menu_idx}/{len(self.menus)} ({menu})")
                    
                    # 메뉴별 진행 로그
                    self.log(f"\n[MENU] [{menu_idx}/{len(self.menus)}] {menu}")
                    
                    result = self.collect_menu(menu, month['start_date'], month['end_date'])
                    results.append(result)
                    
                    time.sleep(1)
                
                if self.running:
                    # 월별 파일 저장
                    filename = self.save_excel(results, month['display'])
                    if filename:
                        saved_files.append(filename)
                    
                    # 월별 통계
                    month_time = time.time() - month_start_time
                    total_estimated = sum(r.get('estimated_count', 0) for r in results)
                    total_actual = sum(r.get('actual_count', 0) for r in results)
                    total_urls = sum(len(r['urls']) for r in results)
                    success_menus = len([r for r in results if r.get('actual_count', 0) > 0])
                    
                    self.log(f"\n[STATS] {month['display']} 완료 결과:")
                    self.log(f"   소요시간: {month_time/60:.1f}분")
                    self.log(f"   성공메뉴: {success_menus}/{len(self.menus)}개")
                    self.log(f"   네이버 추정: {total_estimated:,}개")
                    self.log(f"   실제 확인: {total_actual:,}개")
                    self.log(f"   수집 URL: {total_urls:,}개")
                    self.log(f"   추정 정확도: {(total_actual/total_estimated*100):.1f}%" if total_estimated > 0 else "N/A")
            
            if self.running:
                self.log(f"\n[COMPLETE] 전체 작업 완료!")
                self.log(f"생성된 파일: {len(saved_files)}개")
                for i, file in enumerate(saved_files, 1):
                    self.log(f"   {i}. {file}")
                
                self.status_label.config(text="완료")
                self.current_menu_label.config(text="작업 완료")
                messagebox.showinfo("완료", f"수집 완료!\n파일 {len(saved_files)}개 생성")
            else:
                self.log("작업이 중단되었습니다", "ERROR")
                self.status_label.config(text="중단됨")
                self.current_menu_label.config(text="중단됨")
                
        except Exception as e:
            self.log(f"예상치 못한 오류: {e}", "ERROR")
            messagebox.showerror("오류", str(e))
        finally:
            self.cleanup()
    
    def stop_collection(self):
        """중지"""
        self.running = False
        self.stop_btn.config(state=tk.DISABLED)
        self.log("사용자가 중지를 요청했습니다", "ERROR")
    
    def cleanup(self):
        """정리"""
        self.running = False
        
        if self.driver:
            try:
                self.driver.quit()
                self.log("브라우저 종료", "SUCCESS")
            except:
                pass
            self.driver = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def run(self):
        """실행"""
        def on_closing():
            if self.running:
                if messagebox.askokcancel("종료", "수집 중입니다. 종료하시겠습니까?"):
                    self.cleanup()
                    self.root.destroy()
            else:
                self.cleanup()
                self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

def main():
    if not SELENIUM_OK:
        print("selenium 설치 필요: pip install selenium")
        return
    
    if not EXCEL_OK:
        print("pandas 설치 권장: pip install pandas openpyxl")
    
    if not PIL_OK:
        print("이미지 표시를 위해 Pillow 설치 권장: pip install Pillow")
    
    app = SimpleCollector()
    app.run()

if __name__ == "__main__":
    main()