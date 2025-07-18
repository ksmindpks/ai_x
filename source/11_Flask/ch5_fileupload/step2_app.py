# pip install flask_wtf : 플라스크에서 폼 관리하는 기능
    # CSRF 보호 정책 설정
    # 쉽고 유연한 폼 적용하여 유효성 검증, input 태그 생성
from flask import Flask, render_template #, request.files['file']
from flask_wtf import FlaskForm # 유효성 검사를 위한 form 객체
from flask_wtf.file import FileField, FileRequired # 업로드한 파일 객체
from werkzeug.utils import secure_filename # 업로드한 파일명에 서버에 영향을 미칠 특수문자 빼서 저장
from fileinfo import info # 파일 정보 출력
import os

Uploads_folder = './uploads/'
if not os.path.exists(Uploads_folder):
    os.makedirs(Uploads_folder)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 5  # 업로드 폴더 용량 제한
app.config['SECRET_KEY'] = 'secret!' # CSRF 보호 정책 설정을 하려면 필요, 다르게 uuid.uuid4() 사용도 가능

class FileForm(FlaskForm):
    files = FileField(validators=[FileRequired()]) # 업로드한 파일 객체


@app.route('/', methods=['GET', 'POST'])
def index():
    form = FileForm()
    if form.validate_on_submit(): # 폼 데이터가 유효한지(POST 요청이 유효하게 들어왔는지) 체크
        file = form.files.data # 업로드한 파일 객체
        # 파일명에 서버에 영향을 미칠 특수문자 빼서 저장 
        safe_filename = secure_filename(file.filename)
        file.save(Uploads_folder + safe_filename) # 업로드 폴더에 파일 저장
        ctime, mtime, atime, size = info(safe_filename) # 파일 정보 출력
        return render_template('check.html',
                               fileinfo = {'ctime':ctime,
                                           'size':size})
    else: # GET방식이거나 POST요청이 유효하지 않은 경우
        return render_template('upload.html', form=form) # upload.html 수정
    
if __name__ == '__main__':
    app.run(debug=True)