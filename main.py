import os
import pandas as pd
import time
from ultralytics import YOLO


model = YOLO("runs/detect/train17/weights/best.pt")


while True:
    videos = os.listdir("videos")
    # print(videos)



    if videos:
        for video in videos:
            results = model.predict(f"videos/{video}", stream=True) # Можно запихнуть сразу список файлов
            video_name = ".".join(video.split('.')[:-1])
            coords = {"frame": [], "x": [], "y": []}
            with open(f"videos/{video_name}.txt", 'w+'):
                for frame, result in enumerate(results):
                    result.save_txt(f"videos/{video_name}.txt")
                    for _ in result.boxes:
                        coords["frame"].append(frame)
            with open(f"videos/{video_name}.txt", 'r') as f:
                for data in f:
                    data = data.split()
                    coords["x"].append(data[1])
                    coords["y"].append(data[2])
            os.remove(f"videos/{video_name}.txt")
            # os.remove(f"videos/{video}")
            # print(coords)
            # print(len(coords["frame"]), len(coords["x"]), len(coords["y"]))
            pd.DataFrame(coords).to_csv(f"output/{video_name}.csv", index=False)
    

    time.sleep(10)