import logging
import os
import time
import urllib.request
from itertools import islice
from typing import Dict, List

import cv2
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from constants import DEFAULT_IMAGE_URLS, DEFAULT_LABELS_URL, DEFAULT_MODEL_URL

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable Tensorflow logging

logging.basicConfig(level=logging.INFO)


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

    def order_birds_by_result_score(self, model_raw_output, bird_labels):
        for index, value in np.ndenumerate(model_raw_output):
            bird_index = index[1]
            bird_labels[bird_index]["score"] = value

        return sorted(bird_labels.items(), key=lambda x: x[1]["score"])

    def get_top_n_result(self, top_index, birds_names_with_results_ordered):
        bird_name = birds_names_with_results_ordered[top_index * (-1)][1]["name"]
        bird_score = birds_names_with_results_ordered[top_index * (-1)][1]["score"]
        return bird_name, bird_score

    def main(
        self,
        image_urls: List[str] = DEFAULT_IMAGE_URLS,
    ):
        for index, image_url in enumerate(image_urls):

            logging.info("Loading images")
            image_get_response = urllib.request.urlopen(image_url)
            image_array = np.asarray(
                bytearray(image_get_response.read()), dtype=np.uint8
            )
            logging.info("Changing images")
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
            logging.info("Print results to kubernetes log")
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


if __name__ == "__main__":
    start_time = time.time()
    classifier = BirdClassifier()
    classifier.main()
    print("Time spent: %s" % (time.time() - start_time))
