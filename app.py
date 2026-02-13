import os
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from transformers import AutoProcessor, ChineseCLIPModel, CLIPProcessor, CLIPModel
import base64
import numpy as np
try:
    import pillow_jxl
except ModuleNotFoundError:
    from jxlpy import JXLImagePlugin
try:
    from pillow_heif import HeifImagePlugin
except ModuleNotFoundError:
    pass

# 设备
if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('使用设备', device)

# ==== 初始化 CLIP 模型 ====
timm_model_name = 'vit_large_patch14_clip_224.openai'
timm_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0).to(device)
timm_trans = timm.data.create_transform(**timm.data.resolve_model_data_config(timm_model), is_training=False)

cn_clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14", use_safetensors=False).to(device)
cn_clip_processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14", use_safetensors=False, use_fast=True)

dclip_model = CLIPModel.from_pretrained("OysterQAQ/DanbooruCLIP", use_safetensors=False).to(device)
dclip_processor = CLIPProcessor.from_pretrained("OysterQAQ/DanbooruCLIP", use_safetensors=False)

if (use_search_methods := os.getenv('use_search_methods')) is None or 'ORB' in use_search_methods:
    import cv2
    from python_orb_slam3 import ORBExtractor

    # ==== 初始化 ORB-SLAM3 ====
    orb_extractor = ORBExtractor(
        n_features=500,
        scale_factor=1.2,
        n_levels=8,
        interpolation=cv2.INTER_AREA,
    )

# ==== 特征向量函数 ====
@torch.inference_mode()
def get_timm_vec(img):
    x = timm_trans(img.convert("RGB"))[None, :].to(device)
    feat = timm_model(x)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()

@torch.inference_mode()
def get_cn_clip_vec(img):
    x = cn_clip_processor(images=img, return_tensors="pt").to(device)
    feat = cn_clip_model.get_image_features(**x)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()

@torch.inference_mode()
def get_dclip_vec(img):
    x = dclip_processor(images=[img], return_tensors="pt", padding=True).to(device)
    feat = dclip_model.get_image_features(**x)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()

def get_orb_vec(img):
    cv_im = np.asarray(img.convert("L"))
    _, descriptors = orb_extractor.detectAndCompute(cv_im)
    return descriptors

@torch.inference_mode()
def get_cn_clip_text_vec(text):
    text_features = cn_clip_model.get_text_features(**cn_clip_processor(text=[text], return_tensors="pt", padding=True).to(device))
    text_features = F.normalize(text_features, p=2, dim=1)
    return text_features.cpu().numpy()

@torch.inference_mode()
def get_dclip_text_vec(text):
    text_features = dclip_model.get_text_features(**dclip_processor(text=[text], return_tensors="pt", padding=True).to(device))
    text_features = F.normalize(text_features, p=2, dim=1)
    return text_features.cpu().numpy()


# ==== Gradio 接口函数 ====
def extract_features(model_name, image=None, text=None, return_type='base64'):
    if image is not None:
        # 原有图片特征逻辑
        if model_name == 'vit_large_patch14_clip_224.openai':
            vec = get_timm_vec(image)
        elif model_name == 'chinese-clip-vit-huge-patch14':
            vec = get_cn_clip_vec(image)
        elif model_name == 'DanbooruCLIP':
            vec = get_dclip_vec(image)
        elif model_name == 'ORB-SLAM3':
            vec = get_orb_vec(image)
    elif text is not None:
        # 文本特征逻辑，仅支持 CLIP 模型
        if model_name == 'chinese-clip-vit-huge-patch14':
            vec = get_cn_clip_text_vec(text)
        elif model_name == 'DanbooruCLIP':
            vec = get_dclip_text_vec(text)
        else:
            return "该模型不支持文本输入"

    if return_type == 'nparray':
        return vec
    vec_bytes = vec.tobytes()
    if return_type == 'bytes':
        return vec_bytes
    if return_type == 'base64':
        return base64.b64encode(vec_bytes).decode('utf-8')


if __name__ == "__main__":
    import gradio as gr

    # ==== Gradio App ====
    model_choices = [
        'vit_large_patch14_clip_224.openai',
        'chinese-clip-vit-huge-patch14',
        'DanbooruCLIP',
        'ORB-SLAM3'
    ]

    iface = gr.Interface(
        fn=extract_features,
        inputs=[
            gr.Dropdown(model_choices, label="选择模型"),
            gr.Image(type="pil", label="上传图片 (可选)", format='png'),
            gr.Textbox(label="输入文本 (可选)")
        ],
        outputs=gr.Textbox(label="特征向量（Base64 编码）")
    )
    iface.launch(show_error=True)
