#!/bin/bash

WORKING_DIR="code"


if [ ! -d "$WORKING_DIR" ]; then
    echo "Creating '$WORKING_DIR' directory..."
    mkdir $WORKING_DIR
else
    echo "'$WORKING_DIR' directory already exists."
fi

echo "Setting permissions for 'example' and 'workdir'..."

chmod -R 774 $WORKING_DIR

echo "Building and preparing Docker containers..."
docker-compose up --build --no-start
