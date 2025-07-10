# 플라스크를 사용하기 위한 패키지 설치 : pip install flask
# Flask라는 micro framework를 사용하기 위해
from flask import Flask
from predict import loaded_model, predict_apt_price

application = Flask(__name__) # 어프리케이션 객체 생성

@application.route('/hello')
def handler_function():
    return '<h1>Hello, Flask!</h1>'
# /apt/2005/106/8
@application.route('/apt/<year>/<square>/<floor>')

def aptPredictHandler(year, square, floor):
    answer = predict_apt_price(year, square, floor)
    # return '<h1> 예측금액 : {} </h1>'.format(answer)
    return {
        'year':year,
        'square':square,
        'floor':floor,
        'price':answer
    }
if __name__=='__main__':
    # debug = True : 코드가 변경될 때마다 서버 자동 재시작
    application.run(debug=True)