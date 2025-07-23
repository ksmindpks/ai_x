import cx_Oracle
conn = cx_Oracle.connect('scott',
                         'tiger',
                         '210.121.189.12:1521/xe')
from models import TodoRequest
from typing import List # 타입체크

def get_todos(order) -> List[dict]:
    cursor = conn.cursor()
    if order == 'asc':
        sql = "SELECT * FROM TODO ORDER BY ID"
    else:
        sql = "SELECT * FROM TODO ORDER BY ID DESC"
    cursor.execute(sql)
    result = cursor.fetchall() # 튜플 리스트
    # keys = [desc[0] for desc in cursor.description] # ['id', 'content', 'is_done']
    # todos = [dict(zip(keys, desc)) for desc in cursor.fetchall()]
    cursor.close()
    todos = []
    for row in result:
        todos.append({'id': row[0], 'content': row[1], 'is_done': row[2]})
    return todos

def get_next_id() -> int:
    cursor = conn.cursor()
    sql = "SELECT NVL(MAX(ID), 0) + 1 FROM TODO"
    cursor.execute(sql)
    result = cursor.fetchone() # 튜플 (4,)
    cursor.close()
    return result[0]

def get_todo(id:int) -> dict:
    cursor = conn.cursor()
    sql = "SELECT * FROM TODO WHERE ID = :id"
    cursor.execute(sql, {'id': id})
    result = cursor.fetchone() # 튜플 (1,'바꿀내용', 0)
    cursor.close()
    return {'id': result[0], 'content': result[1], 'is_done': result[2]}

def create_todo(todo: TodoRequest) -> int:
    cursor = conn.cursor()
    sql = "INSERT INTO TODO (ID, CONTENTS, IS_DONE) VALUES (:id, :content, :is_done)"
    cursor.execute(sql,
                   todo.model_dump()) # todo를 dict형태로 변환 {'id': 1, 'content': '바꿀내용', 'is_done': 0}
    conn.commit()
    cursor.close()
    return cursor.rowcount # 추가성공시 1, 실패시 0

def update_todo(todo: TodoRequest) -> int:
    cursor = conn.cursor()
    sql = "UPDATE TODO SET CONTENTS=:content, IS_DONE=:is_done WHERE ID=:id"
    cursor.execute(sql, todo.model_dump())
    conn.commit()
    cursor.close()
    if cursor.rowcount:
        return f'{todo.id}번 {todo.content} 수정 완료'
    else:
        return f'{todo.id}번 수정 실패' # 수정 성공시 1, 실패시 0

def delete_todo(id:int) -> int:
    cursor = conn.cursor()
    sql = "DELETE FROM TODO WHERE ID=:id"
    cursor.execute(sql, {'id': id})
    conn.commit()
    cursor.close()
    if cursor.rowcount: # 삭제성공시 1, 실패시 0
        return f'{id}번 삭제 완료'
    else:
        return f'{id}번 삭제 실패' 

if __name__ == '__main__':
    print('/todos :', get_todos('asc'))
    print('next id :', get_next_id())
    print('/todos/1 :', get_todo(1))
    # todo = TodoRequest(id="307", content="flask 공부", is_done=True)
    # print('/create :', create_todo(todo))
    todo = TodoRequest(id="1", content="바뀐다", is_done=True)
    print('/update/1 :', update_todo(todo))
    print('/delete/1 :', delete_todo(1))

