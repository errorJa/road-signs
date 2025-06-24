import albumentations as A
import cv2


def resize_to_target_size(image_path, target_size: tuple[int, int]):
    """
    Изменяет размер входного изображения на заданную высоту и ширину.
    :param image_path: путь до изображения
    :param target_size: требуемый размер выходного изображения
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    side = min(image.shape[:2])

    transform = A.Compose([
        A.CenterCrop(height=side, width=side),
        A.Resize(height=target_size[0], width=target_size[1])
    ])

    transformed = transform(image=image)
    transformed_image = transformed['image']

    cv2.imwrite(image_path, transformed_image)
