FROM python:3.6-alpine
RUN apk add --update --no-cache tesseract-ocr
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install .
EXPOSE 5000
WORKDIR /usr/src/app/examples/classification
CMD ["flask", "run", "--host", "0.0.0.0"]