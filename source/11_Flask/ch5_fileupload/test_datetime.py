# unix 시간 -> datetime 변환
import datetime
now = datetime.datetime.now().timestamp() # 현재의 unix 시간(70.1.1부터 현재까지 초수)
print(now)
# unix 시간 -> datetime 변환
now_datetime = datetime.datetime.fromtimestamp(now)
print(now_datetime)
print(type(now_datetime))
# datetime 변환 -> unix 시간
unix_time = now_datetime.timestamp()
print(unix_time)