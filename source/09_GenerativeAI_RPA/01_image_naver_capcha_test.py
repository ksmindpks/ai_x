import os
import sys
import urllib.request
from dotenv import load_dotenv

load_dotenv()
client_id = os.getenv('Client_ID')
client_secret = os.getenv('Client_Secret') # 개발자센터에서 발급받은 Client Secret 값
key = "자장면" # 캡차 Key 값
url = "https://openapi.naver.com/v1/captcha/ncaptcha.bin?key=" + key
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    print("캡차 이미지 저장")
    response_body = response.read()
    with open(f'image/captcha{key}.jpg', 'wb') as f:
        f.write(response_body)
else:
    print("Error Code:" + rescode)