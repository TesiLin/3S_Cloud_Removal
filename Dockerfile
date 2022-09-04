FROM python:3.10
WORKDIR /usr/src/app

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install matplotlib --prefer-binary
COPY patch.py patch.py
RUN python patch.py

COPY CloudRemoval .
COPY tif2jpg .
COPY dataset dataset
RUN mkdir /output
WORKDIR /usr/src/app/SpAGAN_total

CMD ["python", "train.py" ]