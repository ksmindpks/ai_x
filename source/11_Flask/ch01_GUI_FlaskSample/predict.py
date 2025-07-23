# 가상환경 만들기:  python -m venv .venv
# 가상환경 들어가기 : .venv\Scripts\activate
# python -m pip install --upgrade pip
# pip install statsmodels
# pip install joblib
# pip install xlwings
# pip install flask

import joblib

loaded_model = joblib.load('../model/ex1_apt_price_regression.joblib')

def predict_apt_price(year, square, floor):
    '건축년도, 전용면적, 층을 입력받아 예측가를 return'
    input_data = [[int(year), int(square), int(floor), 1]]
    pred = round(loaded_model.predict(input_data)[0]*10000)
    return format(pred, ',') + '원'

if __name__=='__main__':
    year = input('몇년 ?')
    square = input('면적(제곱미터) ?')
    floor = input('몇 층 ?')
    print('예측가 :', predict_apt_price(year, square, floor))
