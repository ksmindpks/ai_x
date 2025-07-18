# 특정폴더(uploads/)안의 파일들의 정보를 listup

import os
import datetime

Uploads_folder = './uploads/'

def stamp_to_datetime(stamp):
    return datetime.datetime.fromtimestamp(stamp)

def info(filename):
    ctime = os.path.getctime(Uploads_folder + filename) # 파일의 생성 시간
    mtime = os.path.getmtime(Uploads_folder + filename) # 파일의 수정 시간
    atime = os.path.getatime(Uploads_folder + filename) # 파일의 접근 시간
    size = os.path.getsize(Uploads_folder + filename) # 파일의 크기(byte)
    if size >= 1024 * 1024:
        size = size / (1024 * 1024)
        size = '{:.3f}MB'.format(size)
    elif size >= 1024:
        size = size / 1024
        size = '{:.3f}KB'.format(size)
    else:
        size = '{}B'.format(size)
    return stamp_to_datetime(ctime), stamp_to_datetime(mtime), stamp_to_datetime(atime), size

if __name__ == '__main__':
    filelist = os.listdir(Uploads_folder) # 해당 폴더의 파일이름 목록
    for filename in filelist:
        ctime, mtime, atime, size = info(filename)
        print(filename, ctime, mtime, atime, size)

# filelist = os.listdir(Uploads_folder) # 해당 폴더의 파일이름 목록
# # print(filelist)
# for filename in filelist:
#     ctime = os.path.getctime(Uploads_folder + filename) # 파일의 생성 시간
#     mtime = os.path.getmtime(Uploads_folder + filename) # 파일의 수정 시간
#     atime = os.path.getatime(Uploads_folder + filename) # 파일의 접근 시간
#     size = os.path.getsize(Uploads_folder + filename) # 파일의 크기(byte)
#     print(filename, ctime, mtime, atime, size)

