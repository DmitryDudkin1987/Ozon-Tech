Программа, выполняет следующие шаги для настройки и обучения модели глубокого обучения на наборе данных изображений:

1. Импорт библиотек: Загрузка необходимых библиотек, таких как Matplotlib для визуализации, NumPy для математических операций и TensorFlow для обработки и обучения модели.

2. Загрузка данных:
   - Определяются директории для обучающего и валидационного наборов данных.
   - Устанавливаются параметры BATCH_SIZE (32) и IMG_SIZE (160, 160).
   - Загружается обучающий набор данных с помощью tf.keras.utils.image_dataset_from_directory, который создает батчи изображений и меток.

   В качетсве тренировочных данных используется набор изображений E-CUP: Everything as Code вылоденный в папке data и разбитый на 2 папки OK и NOK.
   Валидация модели проводилась на самостоятельно собранном датасете так же размещенном в папке data.

3. Визуализация данных: 
   - С помощью Matplotlib отображаются 9 случайных изображений из тренировочного набора с соответствующими метками.

4. Очистка данных:
   - С помощью библиотеки python-magic программа проходит по всем файлам в обучающем наборе данных и удаляет файлы, которые не являются изображениями формата JPEG.

5. Создание валидационного и тестового наборов данных:
   - Валидационный набор создается аналогично обучающему.

6. Предварительная обработка и аугментация данных:
   - Настраивается предварительная обработка и аугментация изображений (повороты, отражения).
   - Батчи наборов данных подготавливаются к предварительной обработке с помощью tf.data.AUTOTUNE.

7. Создание модели:
   - Загружается предобученная модель MobileNetV2 без верхнего слоя (include_top=False).
   - Добавляются слои глобального среднего пулинга, полносвязный слой, слой дропаута и выходной слой для классификации.

8. Компиляция модели:
   - Модель компилируется с использованием бинарной кросс-энтропии как функции потерь, Adam как оптимизатор и метрики точности.

9. Файн-тюнинг модели:
   - Указывается количество слоев, которые будут заморожены (не подлежат обучению).
   - Переопределяется оптимизатор и функция потерь и производится повторная компиляция модели для подстройки.

10. Обучение модели:
   - Модель обучается на тренировочном наборе данных с помощью метода fit, включая обратные вызовы для раннего останова валидационной потери.

11. Оценка и предсказание:
   - Извлекается батч изображений из тестового набора, и модель делает прогнозы.
   - Прогнозы преобразуются с применением сигмоиды для получения вероятностей.

12. Визуализация результатов:
   - Отображаются прогнозы модели рядом с соответствующими изображениями для наглядной оценки качества предсказаний.

Программа охватывает полный процесс загрузки, очистки, обработки, обучения и оценки модели глубокого обучения для задачи классификации изображений, используя TensorFlow и Keras.