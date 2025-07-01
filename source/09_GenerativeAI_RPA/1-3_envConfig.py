import os
from dotenv import load_dotenv
from decouple import config
# 방법 1
load_dotenv('.env')
client_id = os.getenv('Client_ID')
print('방법 1:', client_id)
# 방법 2
client_id = config('Client_ID')
print('방법 2:', client_id)

