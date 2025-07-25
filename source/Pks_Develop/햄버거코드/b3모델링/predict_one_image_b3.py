import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from urllib.parse import urlparse
import requests
from io import BytesIO
from model_brand_efficientnet import build_efficientnetb3_model  # 반드시 있어야 함

# ===== 사용자 설정 =====
img_path = r"C:\Users\baby3\OneDrive\바탕 화면\테스트 1.jpg"  # 폴더경로 
model_path = r"C:\Users\Admin\Desktop\b3 모델링\0720_best_efficientnetb3_model.pth" # 경로 설정 확인 필요 
data_dir = r"C:\Users\Admin\Desktop\b3 모델링\test" # 경로 설정 확인 필요 
num_classes = 9

# ===== 디바이스 설정 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 클래스 이름 로딩 =====
class_names = os.listdir(data_dir)
class_names = sorted([d for d in class_names if os.path.isdir(os.path.join(data_dir, d))])

# ===== 전처리 정의 =====
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== 모델 로딩 =====
model = build_efficientnetb3_model(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ===== 이미지 로딩 (로컬 or URL) =====
def load_image(img_path):
    if urlparse(img_path).scheme in ('http', 'https'):
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')
    return img

# ===== 예측 함수: Top-3 결과 출력 =====
def predict_top3(img_path):
    img = load_image(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]  # [C]

    # Top-3 추출
    top3_prob, top3_idx = torch.topk(probs, k=3)
    top3_prob = top3_prob.cpu().numpy()
    top3_idx = top3_idx.cpu().numpy()

    top3_result = []
    for rank, (idx, prob) in enumerate(zip(top3_idx, top3_prob), start=1):
        label = class_names[idx]
        top3_result.append((rank, label, prob * 100))  # 퍼센트 변환

    return top3_result

# ===== 실행 =====
if __name__ == "__main__":
    result = predict_top3(img_path)

    print("\n✅ 예측 결과 (Top-3)")
    for rank, label, prob in result:
        print(f"{rank}위: {label} ({prob:.2f}%)")
