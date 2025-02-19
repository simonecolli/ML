FROM jupyter/minimal-notebook

COPY ./REQUIREMENTS /tmp/REQUIREMENTS
RUN pip install -r /tmp/REQUIREMENTS