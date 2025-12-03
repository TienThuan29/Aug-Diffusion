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


def create_heatmap(scoremap, colormap=cv2.COLORMAP_JET):
    """
    Tạo heatmap thuần túy từ scoremap (không overlay với image)
    Args:
        scoremap: numpy array 2D [H, W] với giá trị normalized [0, 1]
        colormap: OpenCV colormap (mặc định JET)
    Returns:
        heatmap: numpy array [H, W, 3] RGB uint8
    """
    scoremap_uint8 = (scoremap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(scoremap_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def visualize_compound_aug(fileinfos, aug_images, preds, masks, cfg_vis, cfg_reader, reconstructed_images=None):
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

        # --- augmented image: aug_images[i] là tensor [C, H, W] ---
        aug_img = aug_images[i].detach().cpu().numpy()  # [C, H, W]
        aug_img = np.transpose(aug_img, (1, 2, 0))      # -> [H, W, C]
        # nếu ảnh đang ở [0,1] thì scale lên [0,255]
        if aug_img.dtype != np.uint8:
            aug_img = (aug_img * 255.0).clip(0, 255).astype(np.uint8)
        aug_img = cv2.resize(aug_img, (w, h))

        # global normalize theo min/max toàn tập
        pred_clip = np.clip(pred_i, min_score, max_score)
        pred_norm = normalize(pred_clip, max_score, min_score)
        # scoremap_global = apply_ad_scoremap(aug_img, pred_norm)  # Overlay heatmap trên augmented image
        scoremap_global = apply_ad_scoremap(image, pred_norm)
        heatmap_pure = create_heatmap(pred_norm)  # Pure heatmap

        # --- reconstructed image ---
        if reconstructed_images is not None:
            recon_img = reconstructed_images[i].detach().cpu().numpy()  # [C, H, W]
            recon_img = np.transpose(recon_img, (1, 2, 0))  # -> [H, W, C]
            # nếu ảnh đang ở [0,1] thì scale lên [0,255]
            if recon_img.dtype != np.uint8:
                recon_img = (recon_img * 255.0).clip(0, 255).astype(np.uint8)
            recon_img = cv2.resize(recon_img, (w, h))
        else:
            recon_img = None

        # --- ground truth mask ---
        gt_mask = None
        if masks is not None:
            mask_i = masks[i]
            
            # Convert tensor to numpy if needed
            if hasattr(mask_i, 'detach'):
                mask_i = mask_i.detach().cpu().numpy()
            elif hasattr(mask_i, 'numpy'):
                mask_i = mask_i.numpy()
            elif not isinstance(mask_i, np.ndarray):
                mask_i = np.array(mask_i)
            
            # mask có thể là (1, H, W) hoặc (H, W)
            if mask_i.ndim == 3:      # (1, H, W)
                mask_i = mask_i[0]    # -> (H, W)
            elif mask_i.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {mask_i.shape}")
            
            # Check if mask is normalized [0, 1] or [0, 255]
            mask_max = mask_i.max()
            mask_min = mask_i.min()
            mask_mean = mask_i.mean()
            
            if mask_max <= 1.0:
                # Mask is normalized [0, 1], scale to [0, 255]
                gt_mask = (mask_i * 255).astype(np.uint8)
            else:
                # Mask is already [0, 255], just convert to uint8
                gt_mask = np.clip(mask_i, 0, 255).astype(np.uint8)
            
            # Resize mask before converting to RGB
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Convert to RGB format (grayscale to 3 channels)
            gt_mask = np.repeat(gt_mask[..., None], 3, axis=2)  # H x W x 3
            
            # # Debug: print mask statistics (only for first image to avoid spam)
            # if i == 0:
            #     print(f"Mask stats - min: {mask_min:.3f}, max: {mask_max:.3f}, mean: {mask_mean:.3f}, unique values: {len(np.unique(mask_i))}")

        # --- Create visualization: original, augmented, reconstructed, ground truth, heatmap, anomaly map overlay ---
        save_path = os.path.join(save_dir, filename)
        
        # Build stack list based on available images
        stack_list = [image, aug_img]
        
        if recon_img is not None:
            stack_list.append(recon_img)
        
        if gt_mask is not None:
            stack_list.append(gt_mask)
        
        stack_list.extend([heatmap_pure, scoremap_global])

        scoremap = np.vstack(stack_list)              # tất cả đều H x W x 3
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)


# def visualize_compound_aug(fileinfos, aug_images, preds, masks, cfg_vis, cfg_reader, reconstructed_images=None):
#     vis_dir = cfg_vis.save_dir
#     max_score = cfg_vis.get("max_score", None)
#     min_score = cfg_vis.get("min_score", None)

#     # Lấy global min/max từ toàn bộ preds nếu không set sẵn
#     max_score = preds.max() if max_score is None else max_score
#     min_score = preds.min() if min_score is None else min_score

#     image_reader = build_image_reader(cfg_reader)

#     for i, fileinfo in enumerate(fileinfos):
#         clsname = fileinfo["clsname"]
#         filename_full = fileinfo["filename"]
#         filedir, filename = os.path.split(filename_full)
#         _, defename = os.path.split(filedir)
#         save_dir = os.path.join(vis_dir, clsname, defename)
#         os.makedirs(save_dir, exist_ok=True)

#         # --- read original image: H x W x 3, uint8 ---
#         h, w = int(fileinfo["height"]), int(fileinfo["width"])
#         image = image_reader(filename_full)          # giả sử đã là RGB HWC uint8
#         image = cv2.resize(image, (w, h))           # đảm bảo đúng H,W

#         # --- anomaly map pred: lấy đúng 2D map ---
#         pred_i = preds[i]
#         # xử lý khi pred_i có shape (1, H, W) hoặc (H, W)
#         if pred_i.ndim == 3:      # (1, H, W)
#             pred_i = pred_i[0]    # -> (H, W)
#         elif pred_i.ndim != 2:
#             raise ValueError(f"Unexpected pred shape: {pred_i.shape}")

#         pred_i = cv2.resize(pred_i, (w, h))

#         # --- augmented image: aug_images[i] là tensor [C, H, W] ---
#         aug_img = aug_images[i].detach().cpu().numpy()  # [C, H, W]
#         aug_img = np.transpose(aug_img, (1, 2, 0))      # -> [H, W, C]
#         # nếu ảnh đang ở [0,1] thì scale lên [0,255]
#         if aug_img.dtype != np.uint8:
#             aug_img = (aug_img * 255.0).clip(0, 255).astype(np.uint8)
#         aug_img = cv2.resize(aug_img, (w, h))

#         # global normalize theo min/max toàn tập
#         pred_clip = np.clip(pred_i, min_score, max_score)
#         pred_norm = normalize(pred_clip, max_score, min_score)
#         scoremap_global = apply_ad_scoremap(aug_img, pred_norm)  # Overlay heatmap trên augmented image
#         heatmap_pure = create_heatmap(pred_norm)  # Pure heatmap

#         # --- reconstructed image ---
#         if reconstructed_images is not None:
#             recon_img = reconstructed_images[i].detach().cpu().numpy()  # [C, H, W]
#             recon_img = np.transpose(recon_img, (1, 2, 0))  # -> [H, W, C]
#             # nếu ảnh đang ở [0,1] thì scale lên [0,255]
#             if recon_img.dtype != np.uint8:
#                 recon_img = (recon_img * 255.0).clip(0, 255).astype(np.uint8)
#             recon_img = cv2.resize(recon_img, (w, h))
#         else:
#             recon_img = None

#         # --- Create visualization: original, augmented, reconstructed, heatmap, anomaly map overlay ---
#         save_path = os.path.join(save_dir, filename)
        
#         if recon_img is not None:
#             # 5 images: original, augmented, reconstructed, pure heatmap, anomaly map overlay
#             stack_list = [image, aug_img, recon_img, heatmap_pure, scoremap_global]
#         else:
#             # 4 images: original, augmented, pure heatmap, anomaly map overlay
#             stack_list = [image, aug_img, heatmap_pure, scoremap_global]

#         scoremap = np.vstack(stack_list)              # tất cả đều H x W x 3
#         scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(save_path, scoremap)



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
