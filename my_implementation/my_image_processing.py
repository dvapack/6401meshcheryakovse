"""
Модуль my_image_processing.py

Реализация интерфейса IImageProcessing без использования библиотеки OpenCV.

Содержит класс MyImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import interfaces

import numpy as np

from typing import Callable

class MyImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing без использования библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет классическую свёртку изображения с ядром.
        _matrix_convolution(image, kernel): Выполняет свертку изображения с ядром через матричную форму.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray):
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        kern_h, kern_w = kernel.shape
        img_h, img_w= image.shape[0:2]
        out_h, out_w, out_d = img_h - kern_h + 1, img_w - kern_w + 1, image.ndim
        if image.ndim == 2:
            conv_res = np.zeros((out_h, out_w))
            for i in range(out_h):
                for j in range(out_w):
                    conv_res[i, j] = np.sum(image[i:i+kern_h, j:j+kern_w] * kernel)
        else:
            out_d = image.shape[2]
            conv_res = np.zeros((out_h, out_w, out_d))
            for i in range(out_h):
                for j in range(out_w):
                    for d in range(out_d):
                        conv_res[i, j, d] = np.clip(np.sum(image[i:i+kern_h, j:j+kern_w, d] * kernel), 0, 255)
        return conv_res

    def _matrix_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Использует матричную форму свертки.

        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        channels = 3 if image.ndim == 3 else 1
        result = []
        for channel in range(channels):
            channel_image = image if channels == 1 else image[:, :, channel]
            input = np.reshape(channel_image, shape=-1).T
            input_h, input_w = channel_image.shape
            kernel_h, kernel_w = kernel.shape
            output_h, output_w = input_h - kernel_h + 1, input_w - kernel_w + 1
            conv_matrix = np.zeros(shape=(output_h * output_w, input_h * input_w))
            kernel_rows = kernel.shape[0]
            for row in range(conv_matrix.shape[0]):
                i = row // output_w
                j = row % output_w
                start = i * input_w + j
                for kernel_row in range(kernel_rows):
                    pos = start + kernel_row * input_w
                    conv_matrix[row, pos:pos + kernel_w] = kernel[kernel_row]

            channel_result = np.dot(conv_matrix, input.T).reshape(output_h, output_w)
            result.append(channel_result)
        return np.stack(result, axis=-1) if channels == 3 else result[0]

    def _convolution_with_padding(self, image: np.ndarray, kernel: np.ndarray,
                                convolution: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Выполняет свертку с паддингом.
        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        kernel_h, kernel_w = kernel.shape
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        result = convolution(padded_image, kernel)
        return result

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def _gamma_correction(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        image_normalized = image.astype(np.float32) / 255.0
        corrected = np.power(image_normalized, 1.0/gamma)
        return (corrected * 255).astype(np.uint8)


    def edge_detection(self, image: np.ndarray,
                        convolution: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Собеля для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        if convolution is None:
            convolution = self._convolution
        gray = self._rgb_to_grayscale(image)
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        grad_x = convolution(gray, sobel_x)
        grad_y = convolution(gray, sobel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude.astype(np.uint8)

    def corner_detection(self, image: np.ndarray, k_value: float = 0.04, threshold: float = 0.01,
                        convolution: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> np.ndarray:
        """
        Выполняет детектирование углов на изображении с помощью алгоритма Харриса.
        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            k_value (float): Вес следа.
            threshold (float): Порог отклика.

        Returns:
            np.ndarray: Одноканальное изображение с выделенными углами.
        """
        if convolution is None:
            convolution = self._convolution
        if image.ndim == 3:
            gray = self._rgb_to_grayscale(image)
        else:
            gray = image
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        Ix = self._convolution_with_padding(gray, sobel_x, convolution)
        Iy = self._convolution_with_padding(gray, sobel_y, convolution)
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16
        Sxx = self._convolution_with_padding(Ixx, gaussian_kernel, convolution)
        Sxy = self._convolution_with_padding(Ixy, gaussian_kernel, convolution)
        Syy = self._convolution_with_padding(Iyy, gaussian_kernel, convolution)
        det = Sxx * Syy - Sxy**2
        trace = Sxx + Syy
        R = det - k_value * trace**2
        corners = np.array(gray)
        corners[R > threshold * R.max()] = 255

        return corners

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")
