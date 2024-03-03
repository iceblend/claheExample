#https://blog.naver.com/samsjang/220542334453

import cv2
import numpy as np

def histogram_equalization(channel):
    # 히스토그램 계산
    hist, bins = np.histogram(channel.flatten(), 256, [0,256])
    # 누적 히스토그램 생성
    cdf = hist.cumsum()
    # 0이 아닌 최소값을 찾아서 cdf를 정규화합니다.
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    # 정규화된 cdf를 다시 히스토그램 값으로 변경합니다.
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
    # 히스토그램 평활화 적용
    equalized_channel = cdf[channel]
    return equalized_channel

def apply_clahe_to_L_channel(image_lab, clip_limit, grid_size):
    # L 채널을 추출합니다.
    L_channel = image_lab[:,:,0]
    # L 채널에 대해 히스토그램 평활화 적용
    equalized_L = histogram_equalization(L_channel)
    # CLAHE 적용
    clahe_L = equalized_L
    return clahe_L

def apply_clahe_and_convert_back(image, clip_limit, grid_size):
    # 이미지를 BGR에서 LAB로 변환합니다.
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # L 채널에만 CLAHE 적용
    clahe_L = apply_clahe_to_L_channel(image_lab, clip_limit, grid_size)
    # CLAHE 적용된 L 채널을 LAB 이미지의 L 채널에 설정합니다.
    image_lab[:,:,0] = clahe_L
    # LAB 이미지를 BGR로 변환하여 반환합니다.
    result_bgr = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
    return result_bgr



# 입력 이미지 불러오기
input_image = cv2.imread('input_image.jpg')

# CLAHE의 clip limit 및 그리드 크기 설정
clip_limit = 2.0
grid_size = (8, 8)

# CLAHE 적용 및 LAB로 변환 후 다시 BGR로 변환
output_image = apply_clahe_and_convert_back(input_image, clip_limit, grid_size)

# 결과 이미지 저장
cv2.imwrite('output_image.jpg', output_image)