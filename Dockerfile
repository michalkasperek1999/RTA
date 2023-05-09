
FROM python:3.9
#3.11.3-alpine3.17

WORKDIR /app

COPY app.py .
COPY requirements.txt .
COPY model.pkl .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["app.py"]
