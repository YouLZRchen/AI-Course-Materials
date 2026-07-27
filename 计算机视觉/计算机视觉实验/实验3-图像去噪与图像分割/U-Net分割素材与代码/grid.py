import os
import cv2
import numpy as np

# ===================== 配置区 =====================
INPUT_FOLDER = "./output"    # 存放所有分割结果图的文件夹
OUTPUT_IMAGE = "./grid_8x2_merge3.png"  # 合并后大图保存路径
COLS = 2    # 每行2张
ROWS = 8    # 总共8行
# ==================================================

def grid_concat(img_dir, save_path, cols, rows):
    # 读取所有png并按文件名排序
    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    total_need = cols * rows
    if len(img_names) < total_need:
        print(f"警告：文件夹只有{len(img_names)}张图，需要至少{total_need}张才能填满8行2列网格")
    img_names = img_names[:total_need]  # 只取前16张

    # 读取图片
    img_list = []
    for name in img_names:
        path = os.path.join(img_dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"跳过损坏图片：{name}")
            continue
        img_list.append(img)
    if len(img_list) == 0:
        print("无有效图片！")
        return

    # 统一所有图片高度（以第一张高度为基准）
    base_h = img_list[0].shape[0]
    aligned = []
    for im in img_list:
        h, w = im.shape[:2]
        if h != base_h:
            new_w = int(w * base_h / h)
            im = cv2.resize(im, (new_w, base_h))
        aligned.append(im)

    # 按行拼接：每行2张，再把8行上下堆叠
    row_parts = []
    for i in range(0, len(aligned), cols):
        row_slice = aligned[i:i+cols]
        # 一行不足2张时补空白黑图对齐
        if len(row_slice) < cols:
            blank = np.zeros_like(row_slice[0])
            row_slice += [blank] * (cols - len(row_slice))
        row_img = np.hstack(row_slice)
        row_parts.append(row_img)
    final_grid = np.vstack(row_parts)

    # 保存
    cv2.imwrite(save_path, final_grid)
    print(f"拼接完成！网格：{rows}行 × {cols}列，保存至 {save_path}")

if __name__ == "__main__":
    grid_concat(INPUT_FOLDER, OUTPUT_IMAGE, 2, 8)