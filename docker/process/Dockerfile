FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/process

ADD ./requirements_docker.txt .
# ADD ./requirements_docker.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-cache-dir -r requirements_docker.txt


ADD . .