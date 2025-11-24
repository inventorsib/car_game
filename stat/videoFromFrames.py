import cv2
import os
import argparse

def create_video_from_frames(input_folder, output_path, fps=30):
    """
    Создает видео из набора кадров в папке
    
    Args:
        input_folder (str): Путь к папке с кадрами
        output_path (str): Путь для сохранения видео
        fps (int): Количество кадров в секунду
    """
    
    # Получаем список файлов изображений
    images = [img for img in os.listdir(input_folder) 
              if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Сортируем по имени
    
    if not images:
        print("В папке нет изображений!")
        return
    
    # Читаем первое изображение для получения размеров
    first_image_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Создание видео из {len(images)} кадров...")
    
    # Добавляем кадры в видео
    for i, image_name in enumerate(images):
        image_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(image_path)
        video.write(frame)
        
        # Прогресс
        if (i + 1) % 50 == 0:
            print(f"Обработано {i + 1}/{len(images)} кадров")
    
    # Закрываем видео
    video.release()
    print(f"Видео сохранено как: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Создание видео из кадров')
    parser.add_argument('--input', '-i', default=".", help='Папка с кадрами')
    parser.add_argument('--output', '-o', default='output.mp4', help='Выходной файл')
    parser.add_argument('--fps', type=int, default=30, help='FPS видео')
    
    args = parser.parse_args()
    
    create_video_from_frames(args.input, args.output, args.fps)