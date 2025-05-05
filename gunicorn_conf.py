import multiprocessing

# Количество рабочих процессов
workers = multiprocessing.cpu_count() * 2 + 1

# Тип рабочих процессов
worker_class = 'uvicorn.workers.UvicornWorker'

# Привязка к адресу и порту
bind = '0.0.0.0:8000'

timeout = 120

disable_redirect_access_to_syslog = True

max_requests = 1000
max_requests_jitter = 50

loglevel = 'warning'
accesslog = '-'
errorlog = '-'