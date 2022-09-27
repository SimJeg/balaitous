FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y git-core

RUN groupadd -r train && useradd -m --no-log-init -r -g train train

RUN mkdir -p /opt/train /input /output \
    && chown train:train /opt/train /input /output

USER train

WORKDIR /opt/train

ENV PATH="/home/train/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=train:train requirements.txt /opt/train/
RUN python -m pip install --user -r requirements.txt


COPY --chown=train:train main.py /opt/train/
COPY --chown=train:train train.py /opt/train/
COPY --chown=train:train algorithm/ /opt/train/algorithm/
COPY --chown=train:train artifact /opt/train/artifact/

ENTRYPOINT python -m main $0 $@