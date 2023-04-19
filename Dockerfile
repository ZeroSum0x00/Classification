FROM tensorflow/tensorflow:2.9.3-gpu

COPY . /app

RUN apt-get update -y && apt-get install -y apt-transport-https

RUN apt install -y libsm6 libxext6 libxrender1

RUN apt install -y libgl1

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["bash"]