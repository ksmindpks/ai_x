from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index(no=None):
    if request.method == 'POST':
        no = int(request.form.get('no')) # 전달받은 파라미터 값(str)
    return render_template('quiz.html', no=no)

if __name__ == '__main__':
    app.run(debug=True)
