FROM --platform=linux/amd64 python:3.8-slim-buster

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade setuptools pip

RUN apt update
RUN apt-get -y install python-openslide
RUN apt-get -y install git
RUN apt -y install build-essential

RUN pip3 install -r requirements.txt

WORKDIR /opt

COPY . .
EXPOSE 2020
CMD [ "python3", "src/app.py", "Docker" ]