import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO
from celery import Celery, Task
import zipfile
import time
from main_app import main_app_func
import requests

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: tuple, **kwargs: dict) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config.get("CELERY", {}))
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

app = Flask(__name__)

# Папка для сохранения загруженных файлов
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'download'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.config.from_mapping(
    CELERY=dict(
        broker_url="redis://localhost:6379/0",
        result_backend="redis://localhost:6379/0",
    ),
)
celery_app = celery_init_app(app)
socketio = SocketIO(app, message_queue='redis://localhost:6379/0', cors_allowed_origins="*")

# Ограничение на максимальный размер файла (например, 16 Мегабайт)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папку для загрузок, если её еще нет
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@celery_app.task
def process_data(sid):
    # processing
    
    # with open(f"./{DOWNLOAD_FOLDER}/file.txt", 'w') as f:
    #     f.write("aaaaaaaaaaaaaaaa")

    # здесь вызов функции и забрасывание файлов в папку DOWNLOAD_FOLDER
    # main_app_func(data_dir=app.config["UPLOAD_FOLDER"], output_dir=DOWNLOAD_FOLDER)
    response = requests.get("http://localhost:8000/main-app")

    local_socketio = SocketIO(message_queue='redis://localhost:6379/0')
    local_socketio.emit('task_complete', {'result': 'Data processing complete!'}, to=sid)
    return

@socketio.on("start_my_task")
def handle_task_trigger(data):
    from flask import request
    process_data.delay(request.sid)

@app.route('/')
def index():
    # Получаем список всех файлов в папке uploads
    files = os.listdir(DOWNLOAD_FOLDER)
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        return "Ошибка: часть формы с файлом отсутствует", 400
    
    file = request.files['file']
    
    # Если пользователь не выбрал файл
    if file.filename == '':
        return "Ошибка: Файл не выбран", 400
    
    if file:
        # Сохраняем файл в папку uploads под его оригинальным именем
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])
        os.remove(file_path)

        return redirect(url_for('index'))

@app.route('/download')
def download_file():
    filenames = []
    with os.scandir(DOWNLOAD_FOLDER) as entries:
        for i in entries:
            if i.is_file():
                filenames.append(i.path)
    
    with zipfile.ZipFile(f"./{DOWNLOAD_FOLDER}/output.zip", 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        for file in filenames:
            zip.write(file)
    
    # Отправляем файл пользователю из папки uploads
    return send_from_directory(DOWNLOAD_FOLDER, "output.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)