import os
import pathlib

import albumentations as A
import numpy as np
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm


class RandomSolidBackground(ImageOnlyTransform):
    """
    Класс реализует добавление к изображению с прозрачным фоном случайного сплошного фона
    """

    def __init__(self, p=1.0):
        super().__init__(p)

    def apply(self, img, **params):
        # Проверка формата изображения (должен быть RGBA)
        if img.shape[2] != 4:
            return img

        # Отделяем RGB и альфа-канал
        rgb = img[..., :3].astype(np.float32, copy=False)
        alpha = img[..., 3] / 255.0

        # Генерируем случайный цвет фона [R, G, B]
        bg_color = np.random.randint(0, 256, size=(1, 1, 3))

        # Наложение: комбинируем изображение и фон с учётом прозрачности
        result = rgb * alpha[..., None] + bg_color * (1 - alpha[..., None])
        return result.astype(np.uint8, copy=False)


def generate_noisy_images(sample_path: str, output_dir: str, target_size: tuple[int, int], num_images: int = 100):
    """
    Генерирует зашумлённые версии изображения
    :param target_size: размер выходного изображения
    :param sample_path: путь к исходному изображению
    :param output_dir: директория для сохранения результатов
    :param num_images: количество генерируемых изображений
    """
    # Загружаем изображение
    image_height, image_width = target_size
    image = np.array(Image.open(sample_path).convert("RGBA"))

    # Определяем преобразования с помощью Albumentations
    transform = A.Compose([
        A.CenterCrop(height=image_height, width=image_width),
        A.Resize(height=image_height, width=image_width, p=1.0),

        # Геометрические преобразования
        A.RandomOrder([
            A.Affine(scale=(0.7, 1.1), rotate=(-10, 10), border_mode=1, p=0.8),
            A.Perspective(scale=0.1, keep_size=True, fit_output=True, p=0.6),
        ], p=1.0),

        # Добавляем однотонный фон
        RandomSolidBackground(p=1.0),

        # Шумы
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.05, 0.08), p=0.6),
            A.PlasmaShadow(shadow_intensity_range=(0.1, 0.35), roughness=1.5, p=0.4),
            A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.1, 0.2), p=0.4),
        ], p=1.0),

        # Цветовые искажения
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.CLAHE(clip_limit=(1, 8), tile_grid_size=(8, 8), p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
            A.ChromaticAberration(
                primary_distortion_limit=0.05,
                secondary_distortion_limit=0.05,
                mode='random',
                p=0.7
            ),
        ], p=1.0),

        # Размытия
        A.OneOf([
            A.Defocus(radius=(2, 4), p=0.4),
            A.GlassBlur(sigma=0.05, max_delta=1, iterations=1, mode='fast', p=0.4),
            A.MedianBlur(blur_limit=(3, 3), p=0.3),
            A.MotionBlur(blur_limit=(5, 7), angle_range=(0, 0), direction_range=(0.0, 0.0), p=0.6),
        ], p=1.0),

        # Атмосферные эффекты
        A.OneOf([
            A.RandomRain(
                slant_range=(-20, 20),
                drop_length=20,
                blur_value=6,
                brightness_coefficient=0.8,
                rain_type='default',
                p=0.6
            ),
            A.RandomSunFlare(
                flare_roi=(0.03, 0.03, 0.97, 0.12),
                src_radius=300,
                num_flare_circles_range=(5, 9),
                p=0.4
            ),
        ], p=1.0),

        A.PadIfNeeded(min_height=image_height, min_width=image_width, border_mode=1),
    ])

    # Генерация изображений
    for i in tqdm(range(num_images), desc=f"Processing: {os.path.basename(sample_path)}"):
        # Применяем преобразования
        transformed = transform(image=image)
        transformed_image = transformed['image']

        # Сохраняем
        output_name = pathlib.Path(sample_path).with_suffix("").name
        output_path = os.path.join(output_dir, f"{output_name}_{i:04d}.jpg")
        Image.fromarray(transformed_image).save(output_path)


def generate_for_all_classes(samples_dir: str, output_dir: str, target_size: tuple[int, int],
                             num_images_per_class: int):
    """
    Генерация зашумлённых изображений для изображений всех классов
    :param samples_dir: директория с изображениями всех классов
    :param output_dir: директория для сохранения итоговых изображений
    :param target_size: требуемый размер итоговых изображений
    :param num_images_per_class: количество генерируемых изображений для одного класса
    """
    for sample_path in pathlib.Path(samples_dir).iterdir():
        save_dir = os.path.join(output_dir, sample_path.with_suffix("").name)

        if not pathlib.Path(save_dir).exists():
            os.makedirs(save_dir, exist_ok=True)

        generate_noisy_images(sample_path, save_dir, target_size, num_images_per_class)
