# Test Task

Many photographers have been taking images of birds and wondering what kind of bird it actually is.

A bunch of data scientists have been working on a model to help them out.

While the model\* is performing well a lot of corners were cut to get this model to production\** and the service could certainly use some love from a software engineer.

Your task is to:
* Improve service architecture
* Improve service performance
* Improve service maintainability, extensibility and testability

You can change all parts of the code as you see fit, however:
* You are not expected to work on ML model performance
* Model and data have to be fetched online (instead of downloading it to your local machine)

By the end of this task we would like to see, what is a good looking code in your opinion and how much can you optimize latency.

Feel free to play around with the code as much as you like, but in the end we want to see:
* Your vision of nice code
* Code running time including images and model downloading and model inference
* Top 3 results from the model's output per image
* Proper logging for essential and debug info if necessary
* Analyse the bottlenecks in your implementation, and report options for improving upon them

Bonus
* Add CLI and/or web API
    * CLI - I can specify image urls when running the script on commandline, e.g `python classifier.py <url1> <url2>`
    * Web API: HTTP API with an endpoint that accepts one or multiple urls and returns the result. You can use any framework of your choice - FastAPI, Flask, or something else
* Unit tests with Mocked images and model data (possible to run without internet)
* Implement your solution using Docker and Kubernetes for the infrastructure layer. The configuration should scale out: adding machines should reduce latency

The task doesn’t have a fixed time constraint, but we certainly don’t expect you to spend more than 8h.


# Local setup
1) Install Python 3
2) Install requirements `pip install -r requirements.txt`
3) Run the code `python classifier.py`

gl;hf

\* The model:
The sample model is taken from Tensorflow Hub:
https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1

The labels for model outputs can be found here:
https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv

The model has been verified to run with TensorFlow 2.

\** Production: The code was deployed as a python service using Docker with Kubernetes for the infrastructure layer.

# Solution
There is a CLI application that can be run with `python classifier.py`
The application can be passed a list of URLS to images to classify. If no URLs are passed, the application will use a default list of URLs.
The application uses multiprocessing to download the images and classify them in parallel.
It accepts the following arguments:

Options:
  --spawn / --no-spawn            Spawn a new process for each image.
                                  [default: no-spawn]
  --workers INTEGER               Number of workers.  [default: half the available cores]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show the help message and exit.

There is also a webapp. It can be run with `docker-compose build` , `docker-compose push` and `docker-compose up` . It will be available at http://localhost:8000.
The url explorer can be found at http://127.0.0.1:8000/docs.

The webapp uses RQ workers to process the images. The number of workers can be configured in the docker-compose.yml file. Each image is classified in a separate worker. The workers are run in a separate docker container. The webapp is run in a separate docker container. The webapp and the workers communicate via Redis. All the images are tagged to a batch_id which can be used to get the results for the batch.


# Tests
The webapp tests can be run with `python -m pytest . --ignore=data` .
