from math import floor

import cv2
import os


def extract_frames(input_fp, output_dir, interval_sec=0.3):
    """
    Выделяет из видеофайла кадры с интервалом 0.3 секунды
    :param input_fp: путь до видеофайла
    :param output_dir: директория для сохранения выделенных кадров
    :param interval_sec: интервал (в секундах)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_fp)

    if not cap.isOpened():
        print(f'Error: Cannot open {input_fp}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f'Error: Incorrect FPS ({fps})')
        return

    # Рассчитываем интервал в кадрах
    frame_interval = max(1, floor(interval_sec * fps))
    print(f"FPS: {fps:.2f}, frame interval: {frame_interval} frames")

    count = 0
    saved_count = 0
    frame_number = 0

    while True:
        # Устанавливаем позицию для следующего кадра
        next_pos = frame_number + frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
        frame_number = next_pos

        ret, frame = cap.read()

        if not ret:
            break

        # Сохраняем кадр
        filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        count += 1

        # Обновляем счетчик кадров
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    print(f"Saved {saved_count} frames in {output_dir}")
    cap.release()
