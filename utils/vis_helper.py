import os
import cv2
import numpy as np
from data.image_reader import build_image_reader


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    #np_image = np.asarray(image, dtype=np.float)
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualize_compound_aug(fileinfos, aug_images, preds, masks, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)

    # Lấy global min/max từ toàn bộ preds nếu không set sẵn
    max_score = preds.max() if max_score is None else max_score
    min_score = preds.min() if min_score is None else min_score

    image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename_full = fileinfo["filename"]
        filedir, filename = os.path.split(filename_full)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # --- read original image: H x W x 3, uint8 ---
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(filename_full)          # giả sử đã là RGB HWC uint8
        image = cv2.resize(image, (w, h))           # đảm bảo đúng H,W

        # --- anomaly map pred: lấy đúng 2D map ---
        pred_i = preds[i]
        # xử lý khi pred_i có shape (1, H, W) hoặc (H, W)
        if pred_i.ndim == 3:      # (1, H, W)
            pred_i = pred_i[0]    # -> (H, W)
        elif pred_i.ndim != 2:
            raise ValueError(f"Unexpected pred shape: {pred_i.shape}")

        pred_i = cv2.resize(pred_i, (w, h))

        # self-normalized cho phân tích
        scoremap_self = apply_ad_scoremap(image, normalize(pred_i))

        # global normalize theo min/max toàn tập
        pred_clip = np.clip(pred_i, min_score, max_score)
        pred_norm = normalize(pred_clip, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred_norm)

        # --- augmented image: aug_images[i] là tensor [C, H, W] ---
        aug_img = aug_images[i].detach().cpu().numpy()  # [C, H, W]
        aug_img = np.transpose(aug_img, (1, 2, 0))      # -> [H, W, C]
        # nếu ảnh đang ở [0,1] thì scale lên [0,255]
        if aug_img.dtype != np.uint8:
            aug_img = (aug_img * 255.0).clip(0, 255).astype(np.uint8)
        aug_img = cv2.resize(aug_img, (w, h))

        # --- mask (nếu có) ---
        stack_list = None
        if masks is not None:
            mask_i = masks[i]
            # mask có thể là (1, H, W) hoặc (H, W)
            if mask_i.ndim == 3:
                mask_i = mask_i[0]
            elif mask_i.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {mask_i.shape}")

            mask = (mask_i * 255).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_rgb = np.repeat(mask[..., None], 3, axis=2)  # H x W x 3

            save_path = os.path.join(save_dir, filename)
            if mask.sum() == 0:
                # good sample
                stack_list = [image, aug_img, scoremap_global]
            else:
                # defective sample
                stack_list = [image, aug_img, mask_rgb, scoremap_global, scoremap_self]
        else:
            stack_list = [image, aug_img, scoremap_global, scoremap_self]
            save_path = os.path.join(save_dir, filename)

        scoremap = np.vstack(stack_list)              # tất cả đều H x W x 3
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)



# def visualize_compound_aug(fileinfos, aug_images, preds, masks, cfg_vis, cfg_reader):
#     vis_dir = cfg_vis.save_dir
#     max_score = cfg_vis.get("max_score", None)
#     min_score = cfg_vis.get("min_score", None)
#     max_score = preds.max() if not max_score else max_score
#     min_score = preds.min() if not min_score else min_score

#     image_reader = build_image_reader(cfg_reader)

#     for i, fileinfo in enumerate(fileinfos):
#         clsname = fileinfo["clsname"]
#         filename = fileinfo["filename"]
#         filedir, filename = os.path.split(filename)
#         _, defename = os.path.split(filedir)
#         save_dir = os.path.join(vis_dir, clsname, defename)
#         os.makedirs(save_dir, exist_ok=True)

#         # read image
#         h, w = int(fileinfo["height"]), int(fileinfo["width"])
#         image = image_reader(fileinfo["filename"])
#         pred = preds[i][:, :, None].repeat(3, 2)
#         pred = cv2.resize(pred, (w, h))

#         # aug_image shape: [B, C, H, W] need to convert to [H, W, C] for visualize
#         aug_image = aug_images[i].detach().cpu().numpy()  # [C, H, W]
#         aug_image = np.transpose(aug_image, (1, 2, 0))  # [H, W, C]



#         # self normalize just for analysis
#         scoremap_self = apply_ad_scoremap(image, normalize(pred))
#         # global normalize
#         pred = np.clip(pred, min_score, max_score)
#         pred = normalize(pred, max_score, min_score)
#         scoremap_global = apply_ad_scoremap(image, pred)

#         if masks is not None:
#             mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
#             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             save_path = os.path.join(save_dir, filename)
#             if mask.sum() == 0:
#                 scoremap = np.vstack([image, aug_images, scoremap_global])
#             else:
#                 scoremap = np.vstack([image, aug_images, mask, scoremap_global, scoremap_self])
#         else:
#             scoremap = np.vstack([image, aug_images, scoremap_global, scoremap_self])

#         scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(save_path, scoremap)


def visualize_compound(fileinfos, preds, masks, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))
        # global normalize
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        if masks is not None:
            mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(save_dir, filename)
            if mask.sum() == 0:
                scoremap = np.vstack([image, scoremap_global])
            else:
                scoremap = np.vstack([image, mask, scoremap_global, scoremap_self])
        else:
            scoremap = np.vstack([image, scoremap_global, scoremap_self])

        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)


def visualize_single(fileinfos, preds, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # write global normalize image
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        save_path = os.path.join(save_dir, filename)
        scoremap_global = cv2.cvtColor(scoremap_global, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap_global)
