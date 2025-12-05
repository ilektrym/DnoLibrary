import os
import cv2
import numpy as np
import csv
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch
from ultralytics import YOLO
from pathlib import Path

def calculate_center(keypoints):
    """
    Вычисляет центр скелета как точку пересечения диагоналей:
    от левого плеча до правого бедра и от правого плеча до левого бедра.
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :return: Координаты точки пересечения или None, если не найдено
    """
    left_shoulder = keypoints[6]   # Левое плечо
    right_shoulder = keypoints[7]  # Правое плечо
    right_hip = keypoints[12]       # Правое бедро
    left_hip = keypoints[11]        # Левое бедро

    if np.any(np.isnan([left_shoulder, right_shoulder, right_hip, left_hip])):
        return None

    denom = (left_shoulder[0] - right_hip[0]) * (right_shoulder[1] - left_hip[1]) - \
            (left_shoulder[1] - right_hip[1]) * (right_shoulder[0] - left_hip[0])

    px = ((left_shoulder[0]*right_hip[1] - left_shoulder[1]*right_hip[0]) * (right_shoulder[0] - left_hip[0]) -
          (left_shoulder[0] - right_hip[0]) * (right_shoulder[0]*left_hip[1] - right_shoulder[1]*left_hip[0])) / denom

    py = ((left_shoulder[0]*right_hip[1] - left_shoulder[1]*right_hip[0]) * (right_shoulder[1] - left_hip[1]) -
          (left_shoulder[1] - right_hip[1]) * (right_shoulder[0]*left_hip[1] - right_shoulder[1]*left_hip[0])) / denom

    return np.array([px, py])

def calculate_unit_length(keypoints):
    """
    Вычисляет единичный отрезок как среднее расстояний между диагональными ключевыми точками.
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :return: Длина единичного отрезка
    """
    left_shoulder = keypoints[6]
    right_hip = keypoints[7]
    right_shoulder = keypoints[12]
    left_hip = keypoints[11]
    
    if np.any(np.isnan([left_shoulder, right_hip, right_shoulder, left_hip])):
        return None  # Если какие-то из точек отсутствуют, возвращаем None
    
    dist1 = np.linalg.norm(left_shoulder - right_hip)
    dist2 = np.linalg.norm(right_shoulder - left_hip)
    
    return (dist1 + dist2) / 2

def filter_keypoints(keypoints):
    """
    Фильтрует обязательные и необязательные точки.
    Обязательные точки: с 5 по 16 включительно.
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :return: Кортеж (обязательные точки, необязательные точки)
    """
    required_indices = list(range(5, 13))  # Индексы обязательных точек
    required_keypoints = np.array([keypoints[i] for i in required_indices if i < len(keypoints)])
    
    if np.any(np.isnan(required_keypoints)) or len(required_keypoints) < 12:
        return None, None  # Если обязательных точек недостаточно, возвращаем None
    
    return required_keypoints

def calculate_torso_angle(keypoints, shoulder_indices=[5, 6], hip_indices=[11, 12]):
    """
    Вычисляет угол наклона торса относительно вертикали.
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :param shoulder_indices: Индексы точек плеч
    :param hip_indices: Индексы точек бедер
    :return: Угол в радианах или 0, если точки отсутствуют
    """
    # Проверяем, что все необходимые точки для вычисления угла присутствуют
    shoulder_points = keypoints[shoulder_indices]
    hip_points = keypoints[hip_indices]
    
    if np.any(np.isnan(shoulder_points)) or np.any(np.isnan(hip_points)):
        return 0
    
    # Вычисляем середины плеч и бедер
    shoulder_center = np.mean(shoulder_points, axis=0)
    hip_center = np.mean(hip_points, axis=0)
    
    # Вектор торса (от бедер к плечам)
    torso_vector = shoulder_center - hip_center
    
    # Вычисляем угол относительно вертикали
    if np.linalg.norm(torso_vector) == 0:
        return 0
    
    # Нормализуем вектор торса
    torso_vector_normalized = torso_vector / np.linalg.norm(torso_vector)
    
    # Угол между торсом и вертикалью (ось Y)
    angle = np.arctan2(torso_vector_normalized[0], torso_vector_normalized[1])
    
    return angle

def calculate_angle_from_vertical(vector: np.ndarray):
    """
    Вычисляет угол между вектором и вертикальной осью (ось Y) в градусах.
    
    :param vector: Вектор [x, y]
    :return: Угол в градусах от -180 до 180
    """
    # Вертикальный вектор (направленный вниз)
    vertical_vector = np.array([0, -1])
    
    # Вычисляем угол между векторами
    dot_product = np.dot(vector, vertical_vector)
    magnitude_product = np.linalg.norm(vector) * np.linalg.norm(vertical_vector)
    
    if magnitude_product == 0:
        return 0.0
    
    cos_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    if vector[0] < 0:  
        angle_rad = -angle_rad
    
    return angle_rad

def get_vectors_with_lengths_and_angles(keypoints: np.ndarray):
    """
    Возвращает массив векторов между ключевыми точками с их длинами и углами относительно вертикали.
    
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :return: Массив вида [[x, y, длина, угол], ...] или None при ошибке
    """
    required_keypoints = filter_keypoints(keypoints)
    if required_keypoints is None:
        return None
    
    # Определяем пары точек для создания векторов
    vector_pairs = [
        (5, 6),   # плечи
        (11, 12), # бедра
        (5, 11),  # левое плечо - бедро
        (6, 12),  # правое плечо - бедро
        (5, 7),   # левое плечо - локоть
        (7, 9),   # левый локоть - запястье
        (6, 8),   # правое плечо - локоть
        (8, 10),   # правый локоть - запястье
        (11, 13),  # левое бедро - колено
        (12, 14), # правое бедро - колено
        (13, 15), # левое колено - лодыжка
        (14, 16)  # правое колено - лодыжка
    ]
    
    vectors_data = []
    
    for start_idx, end_idx in vector_pairs:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            # Вычисляем вектор
            vector = keypoints[end_idx] - keypoints[start_idx]
            x, y = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]
            
            # Вычисляем длину вектора
            length = np.linalg.norm(vector)
            
            # Вычисляем угол относительно вертикали (в градусах)
            angle = calculate_angle_from_vertical(vector)
            
            # Добавляем в массив: [x, y, длина, угол]
            vectors_data.append([x, y, x2, y2, length, angle])
    
    return np.array(vectors_data)

def normalize_keypoints(keypoints):
    """
    Нормализует кейпоинты относительно нового центра (0, 0), нового масштаба координат
    и угла наклона торса относительно вертикали.
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :return: Массив нормализованных кейпоинтов или None, если обязательные точки отсутствуют
    """
    required_keypoints = filter_keypoints(keypoints)
    if required_keypoints is None:
        return None
    
    center = calculate_center(keypoints)
    if center is None:
        return None
    
    unit_length = calculate_unit_length(keypoints)
    if unit_length is None or unit_length == 0:
        return None
    
    # Применяем нормализацию и поворот
    centered_keypoints = keypoints - center
    normalized_keypoints = centered_keypoints / unit_length
    return normalized_keypoints, [center, unit_length]

def get_normalized_vectors_with_metrics(keypoints: np.ndarray):
    """
    Возвращает нормализованные векторы с длинами и углами + параметры нормализации.
    
    :param keypoints: Массив (N, 2) с координатами кейпоинтов
    :return: (векторы_данные, параметры_нормализации) или None
    """
    # Сначала нормализуем кейпоинты
    normalized_keypoints, norm_params = normalize_keypoints(keypoints)
    if normalized_keypoints is None:
        return None
    
    # Получаем векторы из нормализованных точек
    vectors_data = get_vectors_with_lengths_and_angles(normalized_keypoints)
    
    return vectors_data, norm_params

def process_skeleton(skeleton):
    """
    Обрабатывает последовательность скелетов YOLO, нормализуя их или пропуская, если нет обязательных точек.
    :param skeletons: Список массивов (N, 2) с координатами кейпоинтов
    :return: Список нормализованных скелетов
    """
    for keypoints in skeleton:
        keypoints = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints 
        keypoints = np.array([[np.nan if v is None else v for v in point] for point in keypoints])  
        return get_normalized_vectors_with_metrics(keypoints)[0], get_normalized_vectors_with_metrics(keypoints)[1]
    
def process_skeleton_Norm(skeleton):
    """
    Обрабатывает последовательность скелетов YOLO, нормализуя их или пропуская, если нет обязательных точек.
    :param skeletons: Список массивов (N, 2) с координатами кейпоинтов
    :return: Список нормализованных скелетов
    """
    for keypoints in skeleton:
        keypoints = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints 
        keypoints = np.array([[np.nan if v is None else v for v in point] for point in keypoints])  
        return normalize_keypoints(keypoints)[0], normalize_keypoints(keypoints)[1]

class PoseCompletionNet(nn.Module):
    def __init__(self, input_size=72, hidden_sizes=[512, 1024, 512, 256], output_size=8, dropout_rate=0.3):
        """
        Args:
            input_size: размер входного вектора (72 параметра)
            output_size: размер выходного вектора (24 параметра ног: 48-71)
        """
        super(PoseCompletionNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PoseProcessor:
    """
    Основной класс для обработки поз с автоматической загрузкой моделей
    """
    
    def __init__(self, yolo_model="yolo11n-pose.pt", completion_model_path=None):
        self.yolo_model = YOLO(yolo_model)
        self.completion_model = None
        self.completion_model_path = completion_model_path
        
    def _find_model_file(self, model_path=None):
        """Находит файл модели в различных расположениях"""
        if model_path and os.path.exists(model_path):
            return model_path
            
        # Пробуем разные пути
        possible_paths = [
            Path(__file__).parent / "models" / "NO.pth",  
            Path.cwd() / "models" / "NO.pth",             
            Path.cwd() / "NO.pth",                        
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
                
        raise FileNotFoundError(
            f"Model file NO.pth not found. Checked paths: {possible_paths}\n"
            f"Please place your model file in one of these locations or specify custom path."
        )
    
    def load_completion_model(self, model_path=None):
        """Загружает модель дополнения поз"""
        model_path = self._find_model_file(model_path or self.completion_model_path)
        print(f"Loading completion model from: {model_path}")
        
        model = PoseCompletionNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.completion_model = model
        return model
    
    def process_image(self, image_path):
        """
        Обрабатывает одно изображение и возвращает дополненную позу
        :param image_path: Путь к изображению
        :return: Кортеж (изображение, ключевые точки)
        """
        if self.completion_model is None:
            self.load_completion_model()
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(image_rgb)
        keypoints = results[0].keypoints.xy.cpu().numpy()
        normalized_keypoints = process_skeleton(keypoints)[0]
        norm = process_skeleton_Norm(keypoints)[0]
        
        if normalized_keypoints is None:
            raise ValueError("Could not extract keypoints from image")
        
        with torch.no_grad():
            input_tensor = torch.tensor(normalized_keypoints.flatten().tolist(), 
                                      dtype=torch.float32).unsqueeze(0)
            results = self.completion_model(input_tensor)
            result_numpy = results.cpu().numpy().reshape(-1, 2)
            
        norm[13] = result_numpy[0]
        norm[15] = result_numpy[1]
        norm[14] = result_numpy[2]
        norm[16] = result_numpy[3]

        return [norm*process_skeleton(keypoints)[1][1] + process_skeleton(keypoints)[1][0]]
    
    def plot_keypoints(self, image_path):
        """
        Рисует ключевые точки на изображении (только координаты x, y)
        keypoints_xy: массив формы (num_people, 17, 2)
        """
        image = cv2.imread(image_path)
        keypoints_xy = self.process_image(image_path)
        image_with_keypoints = image.copy()
        
        # Соединения для построения скелета
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # голова
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # руки
            (5, 11), (6, 12), (11, 12),  # туловище
            (11, 13), (13, 15), (12, 14), (14, 16)  # ноги
        ]
        
        # Обрабатываем каждого человека
        for person_keypoints in keypoints_xy:
            color = (0, 255, 0)  # зеленый цвет для скелета
            
            # Рисуем соединения (скелет) - все точки рисуем без проверки confidence
            for start_idx, end_idx in skeleton:
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    start_point = (int(person_keypoints[start_idx][0]), int(person_keypoints[start_idx][1]))
                    end_point = (int(person_keypoints[end_idx][0]), int(person_keypoints[end_idx][1]))
                    
                    cv2.line(image_with_keypoints, start_point, end_point, color, 3)
            
            # Рисуем ключевые точки - все точки рисуем
            for i, point in enumerate(person_keypoints):
                if i < len(person_keypoints):
                    x, y = point[0], point[1]
                    center = (int(x), int(y))
                    cv2.circle(image_with_keypoints, center, 6, (0, 0, 255), -1)  # красные точки
                    cv2.putText(image_with_keypoints, str(i), (center[0]+5, center[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
        result_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)              
        plt.imshow(result_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

