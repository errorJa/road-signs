import argparse
import json
import os
import pathlib

import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
parser = argparse.ArgumentParser(add_help=True)


def select_model(models_dir) -> pathlib.Path | None:
    model_paths = list(pathlib.Path(models_dir).iterdir())
    print('Select model to predict.\nAvailable models:')

    for idx, model_path in enumerate(model_paths):
        print(f'{idx + 1} - {model_path.name}')

    selected_idx = int(input('Enter the number of the model to use: ')) - 1

    if selected_idx < 0 or selected_idx >= len(model_paths):
        print('Invalid index')
        return None
    else:
        return model_paths[selected_idx]


# Группа взаимоисключающих параметров
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    '-g', '--generate-data',
    action="store_true",
    dest='generate_data',
    help='dataset generation'
)
mode_group.add_argument(
    '-t', '--train-classifier',
    action="store_true",
    dest='train_classifier',
    help='training a model for classification'
)
mode_group.add_argument(
    '-s', '--split-video',
    action="store_true",
    dest='split_video',
    help='extracting frames from a video with an interval of 0.3 seconds'
)
mode_group.add_argument(
    '-p', '--predict',
    action="store_true",
    dest='predict',
    help='prediction using a trained model'
)
mode_group.add_argument(
    '-r', '--resize',
    action="store_true",
    dest='rescale',
    help='resizing image to target size'
)

# Остальные параметры
parser.add_argument(
    '-i', '--input',
    action="store",
    dest="input",
    help='path to аn input file'
)
parser.add_argument(
    '--out-dir',
    action="store",
    dest="out_dir",
    help='directory for saving frames from video'
)

if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        config['TARGET_SIZE'] = tuple(config['TARGET_SIZE'])

    if not os.path.exists(config['DATASET_GEN_DIR']):
        os.makedirs(config['DATASET_GEN_DIR'])

    if not os.path.exists(config['PLOTS_DIR']):
        os.makedirs(config['PLOTS_DIR'])

    if not os.path.exists(config['MODELS_DIR']):
        os.makedirs(config['MODELS_DIR'])

    args = parser.parse_args()

    if args.generate_data:
        from tools.data_gen import generate_for_all_classes

        generate_for_all_classes(
            samples_dir=config['SAMPLES_DIR'],
            output_dir=config['DATASET_GEN_DIR'],
            target_size=config['TARGET_SIZE'],
            num_images_per_class=config['NUM_IMAGES_PER_CLASS'],
        )

    if args.split_video:
        if not args.input or not args.out_dir:
            parser.error("--video-split requires both --input and --output-dir")

        from tools.video_split import extract_frames

        extract_frames(input_fp=args.input, output_dir=args.out_dir)

    if args.train_classifier:
        data_gen_classes = os.listdir(config['DATASET_GEN_DIR'])

        if len(data_gen_classes) < 1:
            parser.error('No data generation classes found')
        else:
            from tools.train_classifier import train

            input_sample_path = next(pathlib.Path(config['DATASET_GEN_DIR']).joinpath(data_gen_classes[0]).iterdir())
            image = cv2.imread(input_sample_path, cv2.IMREAD_COLOR_RGB)

            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            train(
                input_shape=image.shape,
                batch_size=config['BATCH_SIZE'],
                epochs=config['EPOCHS'],
                model_version=config['MODEL_VERSION'],
                data_dir=config['DATASET_GEN_DIR'],
                plots_dir=config['PLOTS_DIR'],
                models_dir=config['MODELS_DIR'],
            )

    if args.predict:
        if not args.input:
            parser.error("--predict requires --input")

        from tools.predict_class import predict_class

        selected_model_path = select_model(config['MODELS_DIR'])

        if selected_model_path:
            predict_class(
                model_path=selected_model_path,
                image_path=args.input
            )

    if args.rescale:
        if not args.input:
            parser.error("--rescale requires --input")

        from tools.resize import resize_to_target_size

        resize_to_target_size(
            image_path=args.input,
            target_size=config['TARGET_SIZE']
        )
