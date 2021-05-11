FROM python:3.6-slim
COPY . /

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python3", "test.py"]