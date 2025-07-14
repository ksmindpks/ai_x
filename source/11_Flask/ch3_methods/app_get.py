# python -m venv .venv (가상환경 생성 방법1)
# ctrl+shift+p => select interpreter + 가상환경만들기 
# => .venv로 가상환경 만들기 => 인터프리터경로 입력 => 찾기(python.exe)
# python -m pip install --upgrade pip
# pip install flask

from flask import Flask # 앱 객체
from flask import render_template # html 렌더링
from flask import request # get/post방식으로 파라미터 데이터 받기
from flask import abort # 강제로 예외 발생
from models import Member
from filters import mask_password

app = Flask(__name__)

# 필터링 추가 (str -> str문자갯수만큼 *)
app.template_filter('mask_pw')(mask_password)
# @app.template_filter("mask_pw")
# def mask_password(password):
#     return "*"*len(password)

@app.route('/user/<name>', methods=["GET"]) # /user/hong, methods=['GET']가 default이며 반드시 리스트로 사용
def viewFuction_handlerFunction(name):
    return f'<h1>{name}님 환영합니다. </h1>'

@app.route('/user') # /user?name=hong
def user():
    name = request.args.get('name', '들어온 이름 없음') # get방식 파라미터 값 받기
    if name:
        return f'<h1>전달받은 파라미터 이름 : {name}님</h1>'
    else:
        abort(404)

@app.errorhandler(404) # 404 예외 페이지 처리
def errorhandler(error):
    return render_template('404_pageNotFound.html'), 404

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/join_form", methods=["GET"])
def join_form():
    return render_template('1_onlyget/join.html')

@app.route("/join", methods=["GET"])
def join():
    name = request.args.get('name')
    id = request.args.get('id')
    pw = request.args.get('pw')
    addr = request.args.get('addr')
    member = Member(name, id, pw, addr)
    return render_template("result.html", member=member)

if __name__=='__main__':
    app.run(debug=True, port=80)



class Member:
    def __init__(self, name, id, pw, addr):
        self.name = name
        self.id = id
        self.pw = pw
        self.addr = addr
