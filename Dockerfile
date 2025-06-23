FROM jupyter/minimal-notebook

COPY ./requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt