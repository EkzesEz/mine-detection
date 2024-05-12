from ultralytics import YOLO
import torch
# import torchvision

# CUDA_LAUNCH_BLOCKING=1

# torch.autograd.set_detect_anomaly(True)
def main():
    model = YOLO("yolov8n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    # print(torchvision.__version__)
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    model.to(device)

    results = model.train(data="config.yaml", epochs=300, close_mosaic=300)

if __name__ == '__main__':
    main()