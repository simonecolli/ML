FROM jupyter/minimal-notebook

RUN cp ./requirements-docker.txt /tmp/requirements-docker.txt && pip install -r /tmp/requirements-docker.txt