# 各方法指标对比：双线性插值
import cv2
import os
import re
import math
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def bicubic_super_resolution_paired(input_folder_low_res, input_folder_high_res, scale_factor, output_folder=None):

    total_psnr = 0
    total_ssim = 0

    for filename in tqdm(os.listdir(input_folder_low_res)):

        input_image_low = cv2.imread(os.path.join(input_folder_low_res, filename))
        input_image_high = cv2.imread(os.path.join(input_folder_high_res, re.sub("x\d+", "", filename)))
        output_image = cv2.resize(input_image_low, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_CUBIC)

        if output_folder is not None:
            cv2.imwrite(os.path.join(output_folder, filename), output_image)

        psnr = peak_signal_noise_ratio(input_image_high, output_image)
        ssim = structural_similarity(input_image_high, output_image, multichannel=True)

        total_psnr += psnr
        total_ssim += ssim

    average_psnr = total_psnr / len(os.listdir(input_folder_low_res))
    average_ssim = total_ssim / len(os.listdir(input_folder_low_res))

    return average_psnr, average_ssim


def bicubic_super_resolution_unpaired(input_folder_high_res, scale_factor, output_folder=None):

    total_psnr = 0
    total_ssim = 0

    for filename in tqdm(os.listdir(input_folder_high_res)):

        input_image_high = cv2.imread(os.path.join(input_folder_high_res, filename))

        h_lr = math.floor(input_image_high.shape[0] / scale_factor + 1e-9)
        w_lr = math.floor(input_image_high.shape[1] / scale_factor + 1e-9)
        input_image_high = input_image_high[:round(h_lr * scale_factor), :round(w_lr * scale_factor), :]  # assume round int
        input_image_low = cv2.resize(input_image_high, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)

        output_image = cv2.resize(input_image_low, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_CUBIC)

        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(os.path.join(output_folder, filename), output_image)

        psnr = peak_signal_noise_ratio(input_image_high, output_image)
        ssim = structural_similarity(input_image_high, output_image, multichannel=True)

        total_psnr += psnr
        total_ssim += ssim

    average_psnr = total_psnr / len(os.listdir(input_folder_high_res))
    average_ssim = total_ssim / len(os.listdir(input_folder_high_res))

    print("average psnr of bicubic method is {}".format(round(average_psnr, 4)))
    print("average ssim of bicubic method is {}".format(round(average_ssim, 4)))

    return average_psnr, average_ssim


if __name__ == '__main__':

    bicubic_super_resolution_unpaired("/media/data3/hujt/us_images/us_images", 4)


