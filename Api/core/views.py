import base64
import json
import os.path
import pickle
import time
import tensorflow as tf 

# from django.conf import settings
# from django.core.files.base import ContentFile
# from django.core.files.storage import default_storage
from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
from keras.preprocessing.image import load_img, img_to_array
from rest_framework.views import APIView
import numpy as np
from .detect import detect
from core.models import Plates
from numpy import asarray
# import cv2


class EntryPoint(APIView):
    def get(self, request, *args, **kwargs):
        print(kwargs.get('id'))
        img = img_to_array(Plates.retrieve_image(image_id=kwargs.get('id')))
        r = detect(img)

        print(r)
        return JsonResponse(status=200, data={"id": kwargs.get('id')})
