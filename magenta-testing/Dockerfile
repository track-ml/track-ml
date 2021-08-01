FROM python:3.7.11
ADD . /magenta
WORKDIR /magenta
RUN apt-get update
RUN apt-get -y install build-essential libasound2-dev libjack-dev portaudio19-dev 
RUN pip install -r requirements.txt
CMD [ "python", "./main.py" ] 