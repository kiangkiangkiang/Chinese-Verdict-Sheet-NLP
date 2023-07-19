FROM python:3.9

WORKDIR /code

ENV BASE_DIR=/code
ENV PYTHONPATH=${BASE_DIR}/src
ENV PORT=80

COPY ./requirements.txt ${BASE_DIR}/requirements.txt
COPY ./src ${BASE_DIR}/src

RUN pip install --no-cache-dir --upgrade -r ${BASE_DIR}/requirements.txt
