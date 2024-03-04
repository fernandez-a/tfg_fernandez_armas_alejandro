# Use an official Python base image
FROM python:3.8.18

WORKDIR /usr/app

COPY . /usr/app
COPY ./detr ./detr

ENV PYTHONPATH "${PYTHONPATH}:/usr/detr"
ENV PYTHONPATH "${PYTHONPATH}:/usr/utils"

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install  -r requirements.txt

CMD ["python", "runner.py"]