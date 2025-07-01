# 네이버 api Test (쇼핑검색, 뉴스검색)

import os
import sys
import urllib.request
from dotenv import load_dotenv
load_dotenv()
client_id = os.getenv('Client_ID')
client_secret = os.getenv('Client_Secret')
encText = urllib.parse.quote("포켄스")
media = 'shop' # 혹은 'news'
url = f'https://openapi.naver.com/v1/search/{media}?sort=date&display=2&query={encText}'

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)