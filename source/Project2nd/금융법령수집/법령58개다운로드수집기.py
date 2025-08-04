import time
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_all_financial_laws():
    """금융 법령 전체 다운로드 (1페이지 50개 + 2페이지 8개 = 총 58개)"""
    
    # 다운로드 폴더 설정
    download_path = os.path.abspath("./금융법령_다운로드")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    # Chrome 옵션 설정
    chrome_options = Options()
    
    # 다운로드 설정
    prefs = {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # 봇 탐지 우회 설정
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = None
    total_success = 0
    total_failed = 0
    
    try:
        # Chrome 드라이버 시작
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        wait = WebDriverWait(driver, 20)
        
        logger.info("=== 금융 법령 전체 다운로드 시작 ===")
        
        # 법령 검색 페이지로 이동
        search_url = "https://www.law.go.kr/lsSc.do?section=&menuId=1&subMenuId=15&tabMenuId=81&eventGubun=060101&query=%EA%B8%88%EC%9C%B5"
        driver.get(search_url)
        time.sleep(5)
        
        # 1페이지와 2페이지 순차 처리
        for page_num in [1, 2]:
            logger.info(f"\n=== {page_num}페이지 처리 시작 ===")
            
            # 2페이지로 이동
            if page_num == 2:
                try:
                    page2_link = wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(@onclick, \"movePage('2')\")]"))
                    )
                    driver.execute_script("arguments[0].click();", page2_link)
                    logger.info("2페이지로 이동 성공")
                    time.sleep(3)
                except Exception as e:
                    logger.error(f"2페이지 이동 실패: {e}")
                    break
            
            # 페이지 이동 후 현재 페이지의 법령 링크를 새로 찾기 (중요!)
            try:
                # 표시된(visible) 링크만 찾기
                all_law_links = driver.find_elements(By.XPATH, "//a[contains(@onclick, 'lsViewWideAll') and not(contains(@style, 'display: none'))]")
                
                # 실제로 화면에 보이는 링크만 필터링
                visible_links = []
                for link in all_law_links:
                    try:
                        if link.is_displayed():
                            visible_links.append(link)
                    except:
                        continue
                
                all_law_links = visible_links
                logger.info(f"{page_num}페이지에서 실제 표시된 법령 {len(all_law_links)}개 발견")
                
                if len(all_law_links) == 0:
                    logger.error(f"{page_num}페이지에서 표시된 법령이 없습니다.")
                    continue
                    
            except Exception as e:
                logger.error(f"{page_num}페이지에서 법령 링크를 찾을 수 없습니다: {e}")
                continue
            
            # 현재 페이지의 각 법령 다운로드
            page_success = 0
            page_failed = 0
            
            for i, law_link in enumerate(all_law_links, 1):
                try:
                    global_index = (page_num - 1) * 50 + i
                    
                    logger.info(f"[{global_index}/58] {page_num}페이지 {i}번째 법령 처리 중...")
                    
                    # 법령명 확인
                    law_name = law_link.text.strip()
                    onclick = law_link.get_attribute('onclick')
                    logger.info(f"법령명: {law_name}")
                    
                    # 법령 클릭 (본문 표시)
                    try:
                        driver.execute_script("arguments[0].click();", law_link)
                    except:
                        # 대안: JavaScript 함수 직접 호출
                        params = re.findall(r"'([^']+)'", onclick)
                        if len(params) >= 8:
                            js_code = f"lsViewWideAll('{params[0]}','{params[1]}','{params[2]}',arguments[0],'{params[3]}','{params[4]}','{params[5]}','{params[6]}');"
                            driver.execute_script(js_code, law_link)
                        else:
                            page_failed += 1
                            continue
                    
                    # 본문 로딩 대기 (증가)
                    time.sleep(4)
                    
                    # 서버 과부하 페이지 확인
                    if "사용자가 많아 요청하신 페이지를 정상적으로 제공할 수 없습니다" in driver.page_source:
                        logger.warning("서버 과부하 감지. 30초 대기 후 재시도...")
                        time.sleep(30)
                        
                        # 다시 법령 클릭 시도
                        try:
                            driver.execute_script("arguments[0].click();", law_link)
                            time.sleep(5)
                        except:
                            logger.error(f"재시도 실패: {law_name}")
                            page_failed += 1
                            continue
                    
                    # 본문 저장 버튼 클릭
                    try:
                        save_btn = wait.until(EC.element_to_be_clickable((By.ID, "bdySaveBtn")))
                        driver.execute_script("arguments[0].click;", save_btn)
                        time.sleep(3)  # 팝업 대기 증가
                    except Exception as e:
                        logger.error(f"저장 버튼 클릭 실패: {law_name} - {e}")
                        page_failed += 1
                        continue
                    
                    # DOC 옵션 선택
                    try:
                        doc_radio = wait.until(EC.element_to_be_clickable((By.ID, "FileSaveDoc1")))
                        driver.execute_script("arguments[0].click();", doc_radio)
                        time.sleep(1)
                    except:
                        pass  # DOC 옵션이 없으면 기본 옵션으로 진행
                    
                    # 팝업 저장 버튼 클릭
                    try:
                        popup_save_btn = wait.until(EC.element_to_be_clickable((By.ID, "aBtnOutPutSave")))
                        driver.execute_script("arguments[0].click();", popup_save_btn)
                        time.sleep(3)  # 다운로드 대기 증가
                    except Exception as e:
                        logger.error(f"팝업 저장 실패: {law_name} - {e}")
                        page_failed += 1
                        continue
                    
                    logger.info(f"다운로드 완료: {law_name}")
                    page_success += 1
                    
                    # 다음 법령을 위한 대기 (랜덤 증가)
                    import random
                    wait_time = random.uniform(3, 7)  # 3-7초 랜덤 대기
                    logger.info(f"서버 부하 방지를 위해 {wait_time:.1f}초 대기...")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"법령 다운로드 실패: {law_name} - {e}")
                    page_failed += 1
                    continue
            
            # 페이지 결과
            logger.info(f"{page_num}페이지 완료: 성공 {page_success}개, 실패 {page_failed}개")
            total_success += page_success
            total_failed += page_failed
        
        # 최종 결과
        logger.info(f"\n=== 전체 다운로드 완료 ===")
        logger.info(f"성공: {total_success}개")
        logger.info(f"실패: {total_failed}개")
        logger.info(f"전체: {total_success + total_failed}개")
        
        # 다운로드 파일 확인
        downloaded_files = [f for f in os.listdir(download_path) if f.endswith(('.doc', '.docx', '.hwp', '.pdf'))]
        logger.info(f"다운로드 폴더 내 파일 개수: {len(downloaded_files)}")
        
        print(f"\n모든 법령이 '{download_path}' 폴더에 다운로드되었습니다.")
        
    except Exception as e:
        logger.error(f"전체 프로세스 실패: {e}")
        
    finally:
        if driver:
            driver.quit()
            logger.info("브라우저 종료")

if __name__ == "__main__":
    download_all_financial_laws()