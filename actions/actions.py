# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# This is a simple example for a custom action which utters "Hello World!"

from time import sleep
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from fastai.vision import load_learner, open_image, BytesIO

from PIL import Image
from urllib.request import urlretrieve
from io import BytesIO
import requests


class ActionDetectDisease(Action):

    def name(self) -> Text:
        return "action_detect_disease"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        learn = load_learner("./actions/", 'fastai_learner.pkl')

        img_url = tracker.latest_message["text"]
        img_unique_id = tracker.latest_message["metadata"]["message"]["photo"][-1]["file_unique_id"]

        img_filename = f"./actions/photos/detect_disease/{img_unique_id}.jpg"

        urlretrieve(img_url, img_filename)

        with open(img_filename, "rb") as image:
            img_bytes = image.read()
            img = open_image(BytesIO(img_bytes))
            prediction = str(learn.predict(img)[0])
            print(prediction)
            dispatcher.utter_message(text=prediction)

        return []
