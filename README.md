# DnoLibrary
Библиотека для определения точек ног с чатичным перекрытием.

# Установка

Способ 1: Из GitHub (рекомендуется)
```
pip install "git+https://github.com/ilektrym/DnoLibrary.git#egg=DnoLibrary"
```

Способ 2: Локальная установка
```
# Клонируйте репозиторий
git clone https://github.com/ilektrym/DnoLibrary.git
cd DnoLibrary

# Установите в режиме разработки
pip install -e .
```

# Быстрый старт

Базовый пример
```
from DnoLibrary import PoseProcessor

image = "Путь к файлу"
posePro = PoseProcessor()

#Вывод скилета
print(posePro.process_image(img))

#Вывод скилета на фото
posePro.plot_keypoints(img)
```
