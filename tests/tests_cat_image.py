import unittest
import numpy as np

from lab5.my_implementation import ColorCatImage, GrayscaleCatImage


class TestCatImage(unittest.TestCase):
    """Тесты для класса CatImage"""
    
    def setUp(self):
        self.test_image_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.test_url = "http://example.com/cat.jpg"
        self.test_breed = "TestBreed"
    
    def test_color_image_creation(self):
        """Создание цветного изображения"""
        cat_img = ColorCatImage(self.test_url, self.test_breed, self.test_image_rgb)
        
        self.assertIsInstance(cat_img, ColorCatImage)
        self.assertEqual(cat_img.breed, self.test_breed)
        self.assertEqual(cat_img.url, self.test_url)
        
        self.assertEqual(cat_img.image.shape, (100, 100, 3))
        self.assertTrue(np.array_equal(cat_img.image, self.test_image_rgb))
    
    def test_grayscale_image_creation(self):
        """Создание ч/б изображения"""
        cat_img = GrayscaleCatImage(self.test_url, self.test_breed, self.test_image_rgb)
        
        self.assertIsInstance(cat_img, GrayscaleCatImage)
        self.assertEqual(cat_img.breed, self.test_breed)
        self.assertEqual(cat_img.url, self.test_url)

        self.assertEqual(cat_img.image.shape, (100, 100))
        self.assertEqual(cat_img.image.dtype, np.uint8)
    
    def test_rgb_to_grayscale(self):
        """Тест преобразования RGB в оттенки серого"""
        rgb_image = np.array([
            [[100, 150, 200], [50, 100, 150]],
            [[200, 100, 50], [150, 200, 100]]
        ], dtype=np.uint8)
        
        cat_img = GrayscaleCatImage(self.test_url, self.test_breed, rgb_image)
        self.assertEqual(cat_img.image.shape, (2, 2))
        
        self.assertTrue(np.all(cat_img.image >= 0))
        self.assertTrue(np.all(cat_img.image <= 255))
    
    def test_image_addition(self):
        """Сложение двух изображений"""
        img1 = ColorCatImage(self.test_url, "Breed1", self.test_image_rgb)
        img2 = ColorCatImage(self.test_url, "Breed2", self.test_image_rgb)
        result = img1 + img2
        
        self.assertIsInstance(result, ColorCatImage)
        self.assertEqual(result.breed, "Breed1+Breed2")
    
        expected = self.test_image_rgb.astype(np.int32) * 2
        expected = np.clip(expected, 0, 255).astype(np.uint8)
        self.assertTrue(np.array_equal(result.image, expected))
    
    def test_image_subtraction(self):
        """Вычитание двух изображений"""
        img1 = ColorCatImage(self.test_url, "Breed1", self.test_image_rgb)
        img2 = ColorCatImage(self.test_url, "Breed2", self.test_image_rgb)
        result = img1 - img2
        
        self.assertIsInstance(result, ColorCatImage)
        self.assertEqual(result.breed, "Breed1-Breed2")
        
        expected = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(result.image, expected))
    
    def test_edge_detection_methods(self):
        """Тест методов обнаружения границ"""
        cat_img = ColorCatImage(self.test_url, self.test_breed, self.test_image_rgb)

        custom_edges = cat_img.custom_edge_detection()
        lib_edges = cat_img.library_edge_detection()
        
        self.assertIsInstance(custom_edges, np.ndarray)
        self.assertIsInstance(lib_edges, np.ndarray)
        
        self.assertEqual(custom_edges.ndim, 2)
        self.assertEqual(lib_edges.ndim, 2)

if __name__ == '__main__':
    unittest.main()