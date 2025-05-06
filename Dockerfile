FROM python:3.12-slim   

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /project

RUN pip install --no-cache-dir poetry setuptools
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install ultralytics

COPY pyproject.toml poetry.lock* /project/
COPY gunicorn_conf.py /project/
COPY README.md /project/
COPY ./src/different_approach /project/src

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["python3", "src/main.py", "--weights", "/project/src/crowdhuman.onnx", "--source", "/project/src/detection.mp4", "--save", "--project", "/project/src"]


