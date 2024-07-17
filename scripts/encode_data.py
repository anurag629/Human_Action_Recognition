import os
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py
import json

class UCF50Encoder:
    def __init__(self, root_dir, output_dir, transform=None, sample_rate=10):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_video(self, video_path, class_idx):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                pil_image = Image.fromarray(frame)  # Convert NumPy array to PIL Image
                if self.transform:
                    pil_image = self.transform(pil_image)
                frames.append((np.array(pil_image), class_idx))
            frame_idx += 1
        cap.release()
        return frames

    def save_encoded_data(self, data, batch_idx):
        file_path = os.path.join(self.output_dir, f"data_batch_{batch_idx}.h5")
        with h5py.File(file_path, 'w') as f:
            frames, labels = zip(*data)
            f.create_dataset('frames', data=np.array(frames))
            f.create_dataset('labels', data=np.array(labels))

    def encode_data(self, batch_size=500):
        data = []
        batch_idx = 0
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for video_file in os.listdir(class_dir):
                if video_file.endswith('.avi'):  # Update to handle .mp4 files
                    print(f"Loading {video_file}...")
                    video_path = os.path.join(class_dir, video_file)
                    frames = self.process_video(video_path, self.class_to_idx[cls])
                    data.extend(frames)
                    while len(data) >= batch_size:
                        self.save_encoded_data(data[:batch_size], batch_idx)
                        data = data[batch_size:]
                        batch_idx += 1
        if data:
            self.save_encoded_data(data, batch_idx)
        print("Data encoding completed and saved to disk.")

def main():
    # Load config
    with open('configs/config.json', 'r') as f:
        config = json.load(f)

    # Set up transform based on input_shape from config
    input_shape = config.get("input_shape", [64, 64, 3])  # Default to [64, 64, 3] if not specified
    transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor()
    ])

    encoder = UCF50Encoder(
        root_dir=config['data_dir'],
        output_dir=config['encoded_data_dir'],
        transform=transform,
        sample_rate=config.get('sample_per_vid', 10)
    )
    encoder.encode_data(batch_size=500)

if __name__ == "__main__":
    main()
