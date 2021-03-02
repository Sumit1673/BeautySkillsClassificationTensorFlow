FROM ubuntu:20.04

ENV PATH="root/miniconda3/bin:${PATH}"
ARG PATH="root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y htop python3-dev wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir root/.conda
RUN chmod 0777 Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

RUN rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n beauty python=3.8

COPY . src/
RUN cd src \
    && source activate beauty \
    && pip install -r requirements.txt