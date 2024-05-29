import numpy as np
from PIL import Image
import torch
import cv2
# Tensor to PIL


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class ImageCPostprocessor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "kernel_size": ("INT", {"default": 2})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_postprocess"
    CATEGORY = "Tools"

    def process_image_pil(self, input_image, dilate_kernel_size):
        # 将 PIL 图片转换为带有 alpha 通道的 numpy 数组
        if input_image.mode != 'RGBA':
            print("Provided image does not contain an alpha channel.")
            return None

        original_image = np.array(input_image)
        alpha_channel = original_image[:, :, 3]

        # 创建掩膜，阈值设置为1，区分透明和不透明
        _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

        # 使用 RETR_EXTERNAL 模式找到所有轮廓
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            print("No contours found.")
            return None

        # 遍历每个轮廓，处理
        for contour in contours:
            area = cv2.contourArea(contour)

            # 创建单个区域的掩膜
            temp_mask = np.zeros_like(mask)
            cv2.drawContours(temp_mask, [contour], -1,
                             255, thickness=cv2.FILLED)

            if area < 80:
                # 平均颜色
                # masked_image = cv2.bitwise_and(original_image, original_image, mask=temp_mask)
                # mean_color = cv2.mean(masked_image, mask=temp_mask)[:3]
                # color = tuple(map(int, mean_color)) + (255,)
                # original_image[temp_mask == 255] = color

                # 中值颜色
                masked_image = cv2.bitwise_and(
                    original_image, original_image, mask=temp_mask)
                # 提取被掩膜区域的所有像素值
                masked_pixels = masked_image[temp_mask == 255]

                # 计算每个通道的中值
                if len(masked_pixels) > 0:  # 确保区域内有像素
                    median_color = []
                    for i in range(3):  # 对于 BGR 的每个通道
                        channel_pixels = masked_pixels[:, i]
                        median_value = int(np.median(channel_pixels))
                        median_color.append(median_value)

                    median_color.append(255)  # 添加 alpha 值
                    color = tuple(median_color)
                    # random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)  # 包括 alpha 值
                    original_image[temp_mask == 255] = color
            else:
                masked_image = cv2.bitwise_and(
                    original_image, original_image, mask=temp_mask)
                kernel_size = 3  # 核的大小，决定了放大的像素数
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_image = cv2.dilate(masked_image, kernel, iterations=1)
                original_image[temp_mask ==
                               255] = dilated_image[temp_mask == 255]

        # 重新使用透明通道覆盖透明区域
        original_image[:, :, 3] = mask
        # 将 numpy 数组转回 PIL 图片
        processed_image = Image.fromarray(original_image)

        return processed_image

    def apply_postprocess(self, image, kernel_size):
        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)

                # Apply Fliter
                new_img = self.process_image_pil(pil_image, kernel_size)

                # Output image
                out_image = (pil2tensor(new_img) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:
            pil_image = None
            img = image
            # PIL Image

            pil_image = tensor2pil(img)

            # Apply Fliter
            new_img = self.process_image_pil(pil_image, kernel_size)

            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, )


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageCPostprocessor": ImageCPostprocessor
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCPostprocessor": "Private ImageCPostprocessor"
}
