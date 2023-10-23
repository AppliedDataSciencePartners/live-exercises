# Dockerfile Debugging Exercise

## Introduction

This directory contains the source code for creating a Dockerized Flask based webapp.

The code saved in `./src`, contains a minimal Flask application and a convenience script for running
the application from within a Docker container. 

Don't worry if you haven't used Flask before, the code in `./src` does not need to be amended to be run,
and knowledge of Flask isn't tested as part of this exercise.

## Task

The aim of this exercise is to showcase your Docker knowledge and ability to debug Docker issues, which is a skill
required daily by our Data Scientists and Data Engineers at ADSP.

The Dockerfile in this directory contains a number of bugs, which prevent a Docker image from being built and also
stop the webapp from being run.

Your task is to fix the bugs in the Dockerfile so that the following commands can be run successfully;

Building the image:
```bash
docker build -t adsp-exercise-webapp .
```

Running the webapp:
```bash
docker run --rm -p 5000:5000 adsp-exercise-webapp
```

```bash
$ curl localhost:5000
Hello World!
```

## Hints
During debugging, you may find the following command useful for entering the container rather than running the webapp;
```
docker run --rm -it --entrypoint bash adsp-exercise-webapp
```