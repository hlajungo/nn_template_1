import cv2
import shutil
import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel

# ==========================================
# 初始化 CLIP 模型 (全域載入，避免重複耗時)
# ==========================================
# 載入 Google SigLIP SO400M 模型
print("Loading Google SigLIP SO400M model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/siglip-so400m-patch14-384"
try:
  # 強制只使用本地快取，忽略遠端連線
  clip_model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device)
  clip_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
except Exception as e:
  print(f"本地快取尚未完整，請移除 local_files_only=True 再試一次連網下載。錯誤: {e}")
  # 如果真的漏抓了什麼小檔案，就把 local_files_only=True 拿掉再跑一次

# 定義語意標籤
TEXT_LABELS = [
    # --- Pass (我們想要的：目標植物特徵) ---
    "a close up of a tree leaf",              # 0. 涵蓋：近景、單片大葉子
    "tree canopy and dense foliage",          # 1. 涵蓋：遠景、密集的樹叢、整叢葉子
    "pine needles and branches",              # 2. 涵蓋：松針、細長的針葉特徵
    "tree flowers, blossoms, or colorful buds", # 3. 涵蓋：開花植物 (如羊蹄甲的紫花)、剛發芽的紅葉
    "a close up of a maple leaf",     # 4. 涵蓋：楓葉、掌狀裂葉植物 (新增)

    # --- Fail (我們想踢掉的：戶外常見的非目標物) ---
    "a microscopic close-up of plant veins",   # 顯微鏡級別的植物葉脈特寫 (踢除過度放大的)
    "green grass lawn",                       # 4. 踢除：平坦的綠草地
    "moss and weeds on the ground",           # 5. 踢除：長滿青苔的泥土或地上的碎雜草
    "clear blue sky or cloudy sky",           # 6. 踢除：純天空、雲朵
    "a close up of tree trunk and bark"       # 7. 踢除：只有樹幹或樹皮紋理 (新增)
    ]

# 註冊表
QUALIFICATION_CHECK_REGISTRY = []

# --- 關卡 1: 局部銳利度檢查 ---
def registry_check_focus_local_max(h, s, v, gray, img_path):
  """局部銳利度最大化，解決景深(散景)與粗顆粒高頻雜訊導致的錯殺"""

  # 關鍵修改：將高斯模糊的核(Kernel)加大到 5x5，強制抹平粗顆粒雜訊
  blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

  height, width = blurred_gray.shape
  grid_h, grid_w = height // 4, width // 4
  local_scores = []

  for y in range(0, height - grid_h + 1, grid_h):
    for x in range(0, width - grid_w + 1, grid_w):
      roi = blurred_gray[y:y+grid_h, x:x+grid_w]
      # 針對強力除噪後的圖片計算方差
      score = cv2.Laplacian(roi, cv2.CV_64F).var()
      local_scores.append(score)

  local_scores.sort(reverse=True)
  top_3_score = np.mean(local_scores[:3])

  # 注意：因為使用了更強的 5x5 模糊，整體的方差數值會比之前更低。
  # 門檻可以維持 80，如果發現真實的稍微模糊的葉片被錯殺，可以再降到 60。
  if top_3_score < 120:
    return False, "blurry_local"
  return True, None
QUALIFICATION_CHECK_REGISTRY.append(registry_check_focus_local_max)

# def registry_check_focus_local_max(h, s, v, gray, img_path):
  # """局部銳利度最大化，解決景深(散景)與高頻雜訊導致的錯殺"""

  # # 關鍵修復：先使用 3x3 或 5x5 的高斯模糊抹除像素級噪點，保留真實物理邊緣
  # blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)

  # height, width = blurred_gray.shape
  # grid_h, grid_w = height // 4, width // 4
  # local_scores = []

  # for y in range(0, height - grid_h + 1, grid_h):
    # for x in range(0, width - grid_w + 1, grid_w):
      # roi = blurred_gray[y:y+grid_h, x:x+grid_w]
      # # 針對已經除噪的圖片計算方差
      # score = cv2.Laplacian(roi, cv2.CV_64F).var()
      # local_scores.append(score)

  # local_scores.sort(reverse=True)
  # top_3_score = np.mean(local_scores[:3])

  # # 注意：因為事前做了模糊化，所有真實清晰圖片的得分也會跟著下降
  # # 這裡的閾值需要從原本的 250 調低，建議從 100 或 120 開始測試
  # if top_3_score < 80:
    # return False, "blurry_local"
  # return True, None
# QUALIFICATION_CHECK_REGISTRY.append(registry_check_focus_local_max)

def registry_check_semantics_with_clip(h, s, v, gray, img_path):
  try:
    image = Image.open(img_path).convert('RGB')
    inputs = clip_processor(text=TEXT_LABELS, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
      outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image).cpu().numpy()[0]

    best_match_idx = probs.argmax()
    best_match_label = TEXT_LABELS[best_match_idx]

    if best_match_idx > 4:
      reason = best_match_label.split()[-1]
      return False, f"semantic_{reason}"

    return True, None
  except Exception as e:
    print(f"Semantic 預測失敗: {e}")
    return False, "semantic_error"
QUALIFICATION_CHECK_REGISTRY.append(registry_check_semantics_with_clip)

# --- 主分析函數 ---
def analyze_image_quality(img_path):
  try:
    with Image.open(img_path) as img:
      img = img.convert('RGB')
      img_np = np.array(img)
      hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
      gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

      h, s, v = cv2.split(hsv_img)

      for check_func in QUALIFICATION_CHECK_REGISTRY:
        passed, reason = check_func(h, s, v, gray, img_path)
        if not passed:
          return False, reason

      return True, "pass"
  except Exception:
    return False, "error"


import os
import shutil

# --- 路由與移動腳本 ---
def data_to_review(data_dir, base_review_dir, dest_dir, max_img_num=8000):
  # stats 只紀錄「本次執行」新增的狀態
  stats = {"pass": 0}
  # class_pass_counts 紀錄該類別在 dest_dir 中的「歷史總數 + 本次新增」
  class_pass_counts = {} 

  for root, _, files in os.walk(data_dir):
    # 計算相對路徑作為類別名稱 (例如: 台灣欒樹)
    rel_path = os.path.relpath(root, data_dir)

    # 略過根目錄本身
    if rel_path == '.':
      continue

    print(f"\nWorking on {rel_path}...")

    # 【關鍵修復】初始化該類別的計數器：計算 dest_dir 中現有的合格圖片數量
    if rel_path not in class_pass_counts:
      target_class_dir = os.path.join(dest_dir, rel_path)
      if os.path.exists(target_class_dir):
        # 掃描目標資料夾，只計算圖片檔的數量
        existing_images = [f for f in os.listdir(target_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_pass_counts[rel_path] = len(existing_images)
        print(f" - 發現 {rel_path} 在目標資料夾已有 {class_pass_counts[rel_path]} 張合格圖片。")
      else:
        class_pass_counts[rel_path] = 0

    for file in files:
      # 如果該類別「歷史加上本次」已經達到上限，直接跳出內部迴圈，換下一個植物
      if class_pass_counts[rel_path] >= max_img_num:
        print(f"{rel_path} 總數已達 {max_img_num} 張上限，跳過剩餘圖片。")
        break

      if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

      src_path = os.path.join(root, file)
      is_qualified, reason = analyze_image_quality(src_path)

      if not is_qualified:
        # 不合格：移動到 base_review_dir 供後續檢閱
        specific_review_dir = f"{base_review_dir}_{reason}"
        target_dir = os.path.join(specific_review_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, file)
        shutil.move(src_path, target_path)
        stats[reason] = stats.get(reason, 0) + 1
      else:
        # 合格：移動到 dest_dir (正式訓練集)
        target_dir = os.path.join(dest_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, file)
        shutil.move(src_path, target_path)

        # 更新計數器
        stats["pass"] += 1
        class_pass_counts[rel_path] += 1

  # 印出結算報告
  print("\n本次資料清洗與路由完成！統計結果：")
  for reason, count in stats.items():
    print(f" - 本次新增 {reason.upper()}: {count} 張")

  print("\n各類別在訓練集 (dest_dir) 的最終合格總數：")
  for cls_name, count in class_pass_counts.items():
    print(f" - {cls_name}: {count} 張")


# 使用範例:
if __name__ == '__main__':
  # 將合格的圖片送到 ./data/dataset_train，不合格的送到 ./data/need_review_xxx
  data_to_review(
      data_dir='./data/processed_all',
      base_review_dir='./data/need_review',
      dest_dir='./data/filtered_data',
      max_img_num=5000
      )
