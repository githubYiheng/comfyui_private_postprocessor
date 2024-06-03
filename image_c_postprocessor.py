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
                "kernel_size": ("INT", {"default": 3})
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "success")
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


        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 预定义输入图像最外围必须包含空包围使得内部要被查找的非空区域查找到的面积方向为负值
        need_exp_contours = []
        for contour in contours:
            area = cv2.contourArea(contour, oriented=True)
            if area < 0:
                need_exp_contours.append((contour, area))
        
        if len(need_exp_contours) < 2:
            print("No contours found.")
            return (input_image, False)
        # 遍历每个轮廓，处理
        for contour, area in need_exp_contours:

            # 创建单个区域的掩膜
            temp_mask = np.zeros_like(mask)
            cv2.drawContours(temp_mask, [contour], -1,
                             255, thickness=cv2.FILLED)
            
            
            if 0 > area and -80 < area:
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
            elif -80 >= area:
                masked_image = cv2.bitwise_and(
                    original_image, original_image, mask=temp_mask)
                kernel_size = dilate_kernel_size  # 核的大小，决定了放大的像素数
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_image = cv2.dilate(masked_image, kernel, iterations=1)
                original_image[temp_mask ==
                               255] = dilated_image[temp_mask == 255]

        # 重新使用透明通道覆盖透明区域
        original_image[:, :, 3] = mask
        # 将 numpy 数组转回 PIL 图片
        processed_image = Image.fromarray(original_image)

        return (processed_image, True)

    def apply_postprocess(self, image, kernel_size):
        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)

                # Apply Fliter
                new_img, success = self.process_image_pil(pil_image, kernel_size)

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
            new_img, success = self.process_image_pil(pil_image, kernel_size)

            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, success)

class PrivateImageMask:
    def __init__(self):
        pass
        # def apply_mask_with_opencv(self, input_image, mask_image, mask_gray_threshold):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask_image": ("IMAGE", ),
                "mask_gray_threshold": ("INT", {"default": 127})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_postprocess"
    CATEGORY = "Tools"

    def apply_mask_with_opencv(self, input_image, mask_image, mask_gray_threshold):
        """
        使用OpenCV和NumPy对输入的PIL图像应用掩膜，并返回处理后的PIL图像。
        
        Args:
        input_image (PIL.Image): 被遮罩的PIL图像。
        mask_image (PIL.Image): 用作遮罩的PIL图像。

        Returns:
        PIL.Image: 处理后的PIL图像，掩膜区域变为透明。
        """
        # 将PIL图像转换为OpenCV图像格式
        input_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2BGRA)
        mask_cv = cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2GRAY)

        # 确保掩膜是单通道的
        if len(mask_cv.shape) == 3:
            mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)

        # 将掩膜二值化，阈值可以根据需要调整
        _, binary_mask = cv2.threshold(mask_cv, mask_gray_threshold, 255, cv2.THRESH_BINARY)
        
        # 创建一个新的只有alpha通道的图像，使用掩膜作为alpha通道
        alpha_channel = cv2.bitwise_not(binary_mask)
        b, g, r, a = cv2.split(input_cv)
        bgra = [b, g, r, alpha_channel]

        # 合并通道生成最终图像
        result_cv = cv2.merge(bgra)

        # 将OpenCV图像转换回PIL图像
        result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGRA2RGBA))
        
        return result_pil

    def apply_postprocess(self, image, mask_image, mask_gray_threshold):
        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)

                # Apply Fliter
                new_img = self.apply_mask_with_opencv(pil_image, mask_image, mask_gray_threshold)

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
            new_img = self.apply_mask_with_opencv(pil_image, mask_image, mask_gray_threshold)

            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, )

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageCPostprocessor": ImageCPostprocessor,
    "PrivateImageMask": PrivateImageMask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCPostprocessor": "Private ImageCPostprocessor",
    "PrivateImageMask": "Private Image Mask"
}
