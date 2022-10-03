import logging
import os
import time
import urllib.request
from itertools import islice
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import typer

from constants import DEFAULT_IMAGE_URLS, DEFAULT_LABELS_URL, DEFAULT_MODEL_URL

logging.basicConfig(level=logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable Tensorflow logging


class BirdClassifier:
    def __init__(
        self, model_url: str = DEFAULT_MODEL_URL, labels_url: str = DEFAULT_LABELS_URL
    ):
        logging.info("Loading labels")
        self.labels = self.load_and_cleanup_labels(labels_url)
        logging.info("Loading model")
        self.model = hub.KerasLayer(model_url)

    def load_and_cleanup_labels(self, labels_url: str) -> Dict[int, Dict[str, str]]:
        bird_labels_raw = urllib.request.urlopen(labels_url)
        birds = {}
        for line in islice(bird_labels_raw.readlines(), 1, None):
            bird_label_line = line.decode("utf-8").replace("\n", "")
            bird_id = int(bird_label_line.split(",")[0])
            bird_name = bird_label_line.split(",")[1]
            birds[bird_id] = {"name": bird_name}
        return birds

    def order_birds_by_result_score(
        self, model_raw_output, bird_labels: Dict[int, Dict[str, str]]
    ) -> List[Dict[str, str]]:
        for index, value in np.ndenumerate(model_raw_output):
            bird_index = index[1]
            bird_labels[bird_index]["score"] = value
        return sorted(bird_labels.items(), key=lambda x: x[1]["score"])

    def get_top_n_result(
        self, top_index: int, birds_names_with_results_ordered: List
    ) -> Tuple[str, float]:
        bird_name = birds_names_with_results_ordered[top_index * (-1)][1]["name"]
        bird_score = birds_names_with_results_ordered[top_index * (-1)][1]["score"]
        return bird_name, bird_score

    def main(self, image_url: str):
        logging.info("Loading image")
        try:
            image_get_response = urllib.request.urlopen(image_url)
        except ValueError as e:
            logging.error(f"HTTP Error:{str(e)}")
            logging.error(f"Image URL: {image_url}")
            return []
        image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
        logging.info("Changing image")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        logging.info("Generate tensor")
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        model_raw_output = self.model.call(image_tensor).numpy()
        birds_names_with_results_ordered = self.order_birds_by_result_score(
            model_raw_output, self.labels
        )
        return birds_names_with_results_ordered

    def show_results(self, birds_names_with_results_ordered: List, index: int):
        # Print results to kubernetes log
        print("Run: %s" % int(index + 1))
        bird_name, bird_score = self.get_top_n_result(
            1, birds_names_with_results_ordered
        )
        print('Top match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(
            2, birds_names_with_results_ordered
        )
        print('Second match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = self.get_top_n_result(
            3, birds_names_with_results_ordered
        )
        print('Third match: "%s" with score: %s' % (bird_name, bird_score))
        print("\n")


def main(url: str, index: int):
    classifier = BirdClassifier(
        model_url=DEFAULT_MODEL_URL, labels_url=DEFAULT_LABELS_URL
    )
    logging.info("Waiting for results")
    result = classifier.main(url)
    if not result:
        logging.error("No results")
        return
    classifier.show_results(result, index)


def app(
    urls: Optional[List[str]] = typer.Argument(None, help="Urls of bird images."),
    spawn: bool = typer.Option(False, help="Spawn a new process for each image."),
    workers: int = typer.Option(int(cpu_count() / 2), help="Number of workers."),
):
    start_time = time.time()
    if not urls:
        urls = DEFAULT_IMAGE_URLS
    if spawn:
        if workers > len(urls):
            workers = len(urls)
            logging.info(
                f"Number of workers reduced to {workers} as there \
                are only {len(urls)} images.",
            )
        elif workers < 1:
            workers = 1
            logging.info(
                f"Number of workers reduced to {workers} as minimum worker is one."
            )
        elif workers > cpu_count():
            workers = cpu_count()
            logging.info(
                f"Number of workers reduced to {workers} as \
                there are only {cpu_count()} CPUs."
            )
        pool = Pool(workers)
        logging.info("CPU count: %s" % cpu_count())
        for index, url in enumerate(urls):
            pool.apply_async(
                main,
                (url, index),
            )
        pool.close()
        pool.join()
    else:
        classifier = BirdClassifier(
            model_url=DEFAULT_MODEL_URL, labels_url=DEFAULT_LABELS_URL
        )

        for index, url in enumerate(urls):
            result = classifier.main(url)
            if not result:
                logging.error("No results")
                return
            classifier.show_results(result, index)
    logging.info("Time spent: %s" % (time.time() - start_time))


if __name__ == "__main__":
    typer.run(app)
