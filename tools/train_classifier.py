import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import image_dataset_from_directory
from matplotlib.ticker import MultipleLocator


def train(input_shape: tuple,
          batch_size: int,
          epochs: int,
          model_version: int,
          data_dir: str,
          plots_dir: str,
          models_dir: str):
    # Подготовка датасета
    data_dir = pathlib.Path(data_dir).with_suffix('')

    train_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=input_shape[:2],
        shuffle=True,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=input_shape[:2],
        shuffle=True,
        batch_size=batch_size
    )

    test_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        validation_split=0.1,
        subset="validation",
        seed=321,
        image_size=input_shape[:2],
        shuffle=True,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f'Number of classes: {num_classes}')

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Колбэки для улучшения обучения
    model_checkpoint_path = os.path.join(models_dir, f'RSC-8-v{model_version}-' + 'ep{epoch:02d}.keras')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # Следим за валидационной ошибкой
            patience=5,  # Ждем 5 эпох без улучшений
            restore_best_weights=True,
            min_delta=0.005
        ),
        ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
    ]

    # Создание модели CNN
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1. / 255),  # Нормализация входных данных

        # Блок 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Блок 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Блок 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Блок 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),

        # Выходной слой
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    # Компиляция модели
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Краткое описание модели
    model.summary()

    # Обучение модели
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # Оценка модели на тестовом наборе
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')

    # Сохранение модели
    model_path = os.path.join(models_dir, f'RSC-8-v{model_version}.keras')
    model.save(model_path)

    # Визуализация процесса обучения
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 7.2))

    # График точности
    ax[0].grid()
    ax[0].plot(acc, label='Training Accuracy')
    ax[0].plot(val_acc, label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    ax[0].legend()

    # График потерь
    ax[1].grid()
    ax[1].plot(loss, label='Training Loss')
    ax[1].plot(val_loss, label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1].legend()

    fig_save_path = os.path.join(plots_dir, f'training_history_{model_version}.png')
    plt.savefig(fig_save_path, bbox_inches='tight')
