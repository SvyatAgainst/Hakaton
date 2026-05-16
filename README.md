# Hakaton
Дальноvision

## зависимости для запуска сайта:
```
pip install pandas numpy matplotlib seaborn
pip install "fastapi[standard]"
pip install flask
pip install flask_socketio
pip install celery
pip install zipfile
pip install requests
pip install redis
```

## запуск сайта:
1. запустить backend.py
2. ```fastapi dev main_app.py```
3. ```celery -A backend.celery_app worker --loglevel=info```
