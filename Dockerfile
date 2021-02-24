FROM python:3

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir Cython
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/
COPY data /data/

CMD python . -p
