import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, AsyncMock
from PIL import Image
import io
import asyncio
import shutil

from lab5.my_implementation import CatImageProcessor


class TestCatImageProcessor(unittest.TestCase):
    """Тесты для класса CatImageProcessor"""
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.api_key = "test_api_key"
        self.processor = CatImageProcessor(api_key=self.api_key, output_dir="test_output")
    
    def test_initialization(self):
        """Тест создания объекта"""
        processor = CatImageProcessor(api_key="my_key", output_dir="my_dir")
        self.assertEqual(processor.api_key, "my_key")
        self.assertEqual(processor.output_dir, "my_dir")
        self.assertEqual(processor.base_url, "https://api.thecatapi.com/v1/images/search")
        
    def test_save_image(self):
        """Тест сохранения изображения в файл"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        file_path = os.path.join(self.temp_dir, "test.jpg")
        
        asyncio.run(self.processor.save_image(file_path, test_image))
        
        self.assertTrue(os.path.exists(file_path))
        self.assertGreater(os.path.getsize(file_path), 0)
    
    
    @patch('lab5.my_implementation.cat_image_processor.CatImageProcessor.download_images')
    @patch('lab5.my_implementation.cat_image_processor.CatImageProcessor.process_images_parallel')
    @patch('lab5.my_implementation.cat_image_processor.CatImageProcessor.save_images_async')
    def test_run_async(self, mock_save, mock_process, mock_download):
        """Тест основного пайплайна"""
        mock_download.return_value = []
        mock_process.return_value = []
        mock_save.return_value = None
        
        asyncio.run(self.processor.run_async(limit=5))
        
        mock_download.assert_called_once_with(limit=5)
        mock_process.assert_called_once_with([])
        mock_save.assert_called_once_with([])

class APITestCatImageProcessor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.processor = CatImageProcessor(api_key=self.api_key, output_dir="test_output")
    
    @patch('aiohttp.ClientSession.get')
    async def test_download_images(self, mock_get):
        """Тестирование обращения к API"""
        fake_img = Image.new('RGB', (1, 1), color='red')
        img_bytes = io.BytesIO()
        fake_img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        mock_response_json = AsyncMock()
        mock_response_json.json.return_value = [{
            "url": "http://example.com/cat.jpg",
            "breeds": [{"name": "Test breed"}]
        }]

        mock_response_img = AsyncMock()
        mock_response_img.read.return_value = img_bytes

        mock_get.side_effect = [
            mock_response_json,
            mock_response_img
        ]

        mock_response_json.__aenter__.return_value = mock_response_json
        mock_response_img.__aenter__.return_value = mock_response_img

        results = await self.processor.download_images(limit=1)

        self.assertEqual(len(results), 1)
        data, index, cat_img = results[0]
        self.assertEqual(index, 0)
        self.assertEqual(data["url"], "http://example.com/cat.jpg")
        self.assertEqual(data["breeds"][0]['name'], "Test breed")
        self.assertEqual(cat_img.shape, (1, 1, 3))

        self.assertEqual(mock_get.call_count, 2)
        json_call, img_call = mock_get.call_args_list
        self.assertEqual(json_call[0][0], "https://api.thecatapi.com/v1/images/search")
        self.assertEqual(img_call[0][0], "http://example.com/cat.jpg")

if __name__ == '__main__':
    unittest.main()