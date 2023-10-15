FROM python:3.9.13
COPY . .
RUN pip install -r requirements.txt
CMD ["python","wsgi.py"]