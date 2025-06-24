import os

import keras
import numpy as np


def predict_class(model_path: str | os.PathLike, image_path: str):
    cls_idx = {
        0: 'danger',
        1: 'end_restricted_area',
        2: 'entry_prohibited',
        3: 'motor_vehicles_prohibited',
        4: 'overtaking_prohibited',
        5: 'parking_not_allowed',
        6: 'pedestrian_traffic_prohibited',
        7: 'stopping_not_allowed',
        8: 'traffic_prohibited',
        9: 'uturn_prohibited',
    }
    cls_idx_ru = {
        0: 'Знак 3.17.2. Опасность',
        1: 'Знак 3.31. Конец зоны всех ограничений',
        2: 'Знак 3.1. Въезд запрещен',
        3: 'Знак 3.3. Движение механических транспортных средств запрещено',
        4: 'Знак 3.20. Обгон запрещен',
        5: 'Знак 3.28. Стоянка запрещена',
        6: 'Знак 3.10. Движение пешеходов запрещено',
        7: 'Знак 3.27. Остановка запрещена',
        8: 'Знак 3.2. Движение запрещено',
        9: 'Знак 3.19. Разворот запрещен',
    }

    model = keras.models.load_model(model_path)
    image = keras.utils.load_img(image_path)
    image_np = keras.utils.img_to_array(image, dtype="float32")

    predictions = model.predict(np.array([image_np]), verbose=0)
    print(f'Predict result: {cls_idx[np.argmax(predictions)]} - {cls_idx_ru[np.argmax(predictions)]}')
