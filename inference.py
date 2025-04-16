import os
import glob
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import GeneratorResNet


def load_model(weights_path, device):
    model = GeneratorResNet(input_shape=(3, 256, 256), num_residual_blocks=9).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0)


def postprocess_tensor(tensor):
    tensor = 0.5 * (tensor + 1.0)
    tensor = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    return tensor


def run_inference_on_folder(input_folder, output_folder, weights_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)

    model = load_model(weights_path, device)

    image_paths = glob.glob(os.path.join(input_folder, "*.*"))

    for img_path in image_paths:
        input_img = preprocess_image(img_path).to(device)

        with torch.no_grad():
            output = model(input_img)

        result = postprocess_tensor(output)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        plt.imsave(save_path, result)
        print(f"Сохранено: {save_path}")


# Использование
run_inference_on_folder(
    input_folder="input_images",
    output_folder="output_images",
    weights_path="saved_models/AIvazovsky/G_BA_final.pth"
)