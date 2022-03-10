from django.db import models
import cv2
import numpy as np


class Plates(models.Model):
    content = models.FileField(blank=False)
    name = models.CharField(max_length=255, blank=True)

    class Meta:
        managed = False  # remove this line
        db_table = 'plates'

    def __str__(self):
        return f'{self.id}'

    @classmethod
    def get_plate_by_id(cls, id: str) -> object:
        return cls.objects.filter(id=id).first()

    @classmethod
    def retrieve_image(cls, image_id):
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(cls.objects.filter(id=image_id).first().content, dtype=np.uint8), flags=1), cv2.COLOR_BGR2RGB)
