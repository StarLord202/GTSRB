import argparse
import os
import glob
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet34

IND_TO_CLASS_GTRSB = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "Crossed 80 limit",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "Two cars red and black",
    10: "Big red small black cars",
    11: "Triangle with crossbar",
    12: "White yellow inside rectangle",
    13: "Upside down triangle white inside",
    14: "STOP",
    15: "White inside circle",
    16: "Circle truck inside",
    17: "NO WAY SIGN",
    18: "! SIGN",
    19: "Triangle turn left",
    20: "Triangle turn right",
    21: "Triangle sneak road",
    22: "Triangle with bumps",
    23: "Slippery road",
    24: "Triangle straight and curved road",
    25: "Triangle workman",
    26: "Traffic light",
    27: "Pedestrian",
    28: "People run",
    29: "Triangle bicycle",
    30: "Triangle snow",
    31: "Triangle deer",
    32: "Crossed circle",
    33: "Turn right",
    34: "Turn left",
    35: "Go straight",
    36: "Straight or right",
    37: "Straight or left",
    38: "Crossed right",
    39: "Crossed left",
    40: "Roundabout",
    41: "Crossed cars",
    42: "Crossed big and small car"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(device)


def get_model():
    model = resnet34()
    model.fc  = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512,43)
    )
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    model.to(device)
    return model

def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing image samples')
    parser.add_argument('--scan_mode', choices=["img", "sub"], default="img", help="Choose img if u want scan directory only for images, choose sub if u want to scan dir and it's subdirs for images ")
    args = parser.parse_args()

    if args.scan_mode == "img":
        image_paths = glob.glob(os.path.join(args.image_dir, '*.png')) + glob.glob(
            os.path.join(args.image_dir, '*.jpg')) + glob.glob(os.path.join(args.image_dir, '*.jpeg'))

    else:
        image_paths = get_image_paths(args.image_dir)

    model = get_model()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    print(f"total {len(image_paths)} images found")

    for image_path in image_paths:
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output.data, 1)
        symbol = IND_TO_CLASS_GTRSB[predicted.item()]

        print(f"{symbol}, {image_path}")

if __name__ == '__main__':
    main()