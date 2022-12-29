FROM python:3.8-buster

RUN apt-get update && \
    apt-get install -y libxaw7-dev

RUN wget https://sourceforge.net/projects/ngspice/files/ng-spice-rework/37/ngspice-37.tar.gz/download -O ngspice-37.tar.gz && \
    tar -zxvf ngspice-37.tar.gz && \
    cd ngspice-37 && \
    mkdir release && \
    cd release && \
    ../configure  --with-x --with-readline=yes --disable-debug && \
    make && \
    make install

RUN mkdir ./Circuit-Synthesis

WORKDIR ./Circuit-Synthesis

COPY requirements.txt requirements.txt

RUN pip install --no-cache -r requirements.txt

COPY . .

RUN python setup.py

ENTRYPOINT ["python", "main.py"]

CMD ["--path", "./config/train_config.yaml"]