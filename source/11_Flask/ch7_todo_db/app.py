# pip install flask
# pip install pydantic
# pip install cx_Oracle

from flask import Flask, request, render_template
from flask import redirect, url_for, abort # redirect, 강제로 예외 발생
from flask import session # 로그인/로그아웃
from models import TodoRequest
from repository import get_todos, get_next_id, get_todo
from repository import create_todo, update_todo, delete_todo

app = Flask(__name__)
# app.secret_key = '안녕하세요'
app.config['SECRET_KEY'] = 'abc123'

@app.route('/')
def index():
    '로그인 성공 로직(세션에 로그인 한 사람의 정보 넣기) 후 /todos로 리다이렉트'
    session['user_id'] = 'hong'
    session['user_name'] = '홍길동' 
    # return redirect('/todos') # /todos(GET)로 요청경로로 
    return redirect(url_for('todos')) # /todos 함수로

@app.route('/logout')
def logout():
    '세션의 값 날리고 /todos로 리다이렉트'
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('todos')) # /todos 로 리다이렉트

@app.route('/todos') # 전체 목록 보기
def todos():
    'todo_data를 list로 변환하여 렌더링'

    order = request.args.get('order', 'asc') # 정렬 순서: asc, desc
    ret = get_todos(order)
    next_id = get_next_id()
    return render_template('todo/todos.html',
                           todos=ret,
                           next_id=next_id,
                           order=order)

@app.route('/todos/<int:id>') # 해당 id 상세보기
def todo(id):
    '해당 id의 todo_data를 html로 렌더링'
    todo = get_todo(id)
    if todo:
        return render_template('todo/todo.html', todo=todo)
    return abort(404, description='해당 id의 할일이 없습니다.')

@app.route('/create', methods=['POST'])
def create():
    '새로운 할일(request.form)을 추가하는 로직'

    todo = TodoRequest(**request.form.to_dict())
    create_todo(todo)
    return redirect(url_for('todos')) # /todos 로 (GET방식) 리다이렉트

@app.route('/update/<int:id>', methods=['GET']) # 수정할 수 있는 페이지로 가기
def get_update(id):
    return render_template('todo/update.html', todo=get_todo(id))

@app.route('/update/<int:id>/<string:content>/<string:is_done>', methods=['PUT']) # 수정할 수 있는 페이지로 가기
def update(id, content, is_done):
    todo = TodoRequest(id=id, content=content, is_done=is_done)
    return update_todo(todo) 

@app.route('/delete/<int:id>', methods=['DELETE'])
def delete(id):
    return delete_todo(id)

@app.errorhandler(404)
def not_found(error):
    return render_template('page_not_found.html', error=error), 404

if __name__ == '__main__':
    app.run(debug=True)