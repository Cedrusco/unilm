FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get update \
    && apt-get install tesseract-ocr -y \
    && apt-get clean \
    && apt-get autoremove
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install .
EXPOSE 5000
WORKDIR /usr/src/app/examples/classification
ENTRYPOINT python3 app.py