import os
import re
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR

# 提取信息并保存bounding box图像和信息JSON的函数
def extract_label_info_and_save_bbox(image_path, output_folder):
    ocr = PaddleOCR()
    result = ocr.ocr(image_path, cls=True)

    # 提取所有文本，保留换行
    lines = [line[1][0] for sublist in result for line in sublist]
    full_text = "\n".join(lines)

    info = {
        "manufacturer": None,
        "serial_number": None,
        "part_number": None,
        "model": None,
        "address": None,
        "contact": None
    }

    patterns = {
        "serial_number": [
            r"(?:S\/N|SN|Serial\s*Number|ID)[:\s]*([A-Za-z0-9.-]+)",
            r"S\/N\s*(\d+)",
            r"SN[:]?\s*(\d+)",
            r"ID[:]?\s*([A-Za-z0-9.-]+)"
        ],
        "part_number": [
            r"(?:Part\s*No|Part\s*Number|P\/N)[:\s]*([A-Za-z0-9-]+)",
            r"P\/N\s*(\d+)",
            r"Part\s*:\s*(\d+)"
        ],
        "model": [
            r"(?:Model|Type|Product)[:\s]*([A-Za-z0-9-]+)",
            r"MODEL\s*([A-Z0-9]+)"
        ],
        "manufacturer": [
            r"^([A-Z][A-Za-z\s-]+(?:GmbH|Inc|LLC|Ltd|Corp)?)\b",
            r"Manufacturer:\s*([A-Z][A-Za-z\s-]+)"
        ],
        "address": [
            r"([A-Za-zäöüßÄÖÜ\s.-]+\s\d+\n\d+\s[\w\s,]+,\s*\w+)",
            r"(\d+\s[A-Za-z\s.]+\n[A-Za-z]+,\s[A-Z]{2}\s\d+)"
        ],
        "contact": [
            r"([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})",
            r"(?:Contact|Email)[:\s]*([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})"
        ]
    }

    for field in patterns:
        for pattern in patterns[field]:
            match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip().replace('\n', ' ').replace('\r', '')
                info[field] = value
                break

    if not info["manufacturer"] and len(lines) > 0:
        if re.match(r"^[A-Z][A-Z\s-]+$", lines[0]):
            info["manufacturer"] = lines[0].replace('\n', ' ').strip()

    if not info["model"]:
        for line in lines:
            if (not re.match(r"^(S\/N|Part|P\/N|Serial)", line, re.IGNORECASE) and
                len(line.split()) <= 5 and
                not re.match(r"^\d+$", line)):
                info["model"] = line.replace('\n', ' ').strip()
                break

    clean_info = {k: v.replace('\n', ' ').strip() for k, v in info.items() if v is not None}

    # === 绘制边框 ===
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for sublist in result:
        for line in sublist:
            box = [tuple(map(int, point)) for point in line[0]]
            text = line[1][0]
            cv2.polylines(image, [np.array(box)], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(image, text, (box[0][0], box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # === 保存图像和JSON ===
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_folder, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    output_image_path = os.path.join(image_output_dir, f"{image_name}_bbox.jpg")
    output_json_path = os.path.join(image_output_dir, f"{image_name}_info.json")

    cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_info, f, ensure_ascii=False, indent=4)

    return clean_info


# 处理整个文件夹
def process_images_in_folder(folder_path, output_folder):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    all_info = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")

        info = extract_label_info_and_save_bbox(image_path, output_folder)
        all_info.append({
            "image": image_file,
            "extracted_info": info
        })

    return all_info


# === 设置路径 ===
input_folder = r"E:\OCR\inputs"
output_folder = r"E:\OCR\result"

os.makedirs(output_folder, exist_ok=True)
extracted_data = process_images_in_folder(input_folder, output_folder)

# 输出总览
for data in extracted_data:
    print(f"Image: {data['image']}")
    print(f"Extracted Info: {data['extracted_info']}\n")
