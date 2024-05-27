import io

import torch
from torch import nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alstjd2627.github.io/*"],  # 또는 React 앱의 도메인을 명시
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

my_dict = {
    0: 'Actinic keratoses',
    1: 'Dermatofibroma',
    2: 'Benign keratosis-like lesions ',
    3: 'Basal cell carcinoma',
    4: 'Melanocytic nevi',
    5: 'Vascular lesions',
    6: 'dermatofibroma',
    7: 'acne_1',
    8: 'acne_2',
    9: 'acne_3',
    10: 'normal'
}


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b3')
        self.model._fc = nn.Linear(self.model._fc.in_features, 11)  # 클래스 개수에 맞게 조정

    def forward(self, x):
        return self.model(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_state_dict = torch.load("saved_model.pth", map_location=device)

model = CustomModel()
model.load_state_dict(model_state_dict, strict=False)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(300),  # EfficientNet-B3의 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # EfficientNet-B3에 맞는 정규화 값
])


# @app.get("/")
# def main():
#     return FileResponse('index.html')

@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    predicted_class = torch.argmax(output, dim=1).item()
    return {"predicted_class": my_dict[predicted_class]}


@app.get("/")
def get_ok():
    return {"success": 200}
