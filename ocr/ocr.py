#&&
import cv2
import numpy as np
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
result_chars = ""

# 1) ori img 호출
img_ori = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
height, width, channel = img_ori.shape
##############
#cv2.imshow('Image Basic',  img_ori)
#cv2.waitKey(0)
############

# 2) gray로 변조
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
##############
#cv2.imshow('Image Basic',  img_gray)
#cv2.waitKey(0)
############

# 3) blur처리
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
##############
#cv2.imshow('Image Basic',  img_blurred)
#cv2.waitKey(0)
############

# 4) thresh처리
img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
##############
#cv2.imshow('Image Basic',  img_thresh)
#cv2.waitKey(0)
############

# 5) contours 적용
contours, _ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

#cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
##############
#cv2.imshow('Image Basic',  temp_result)
#cv2.waitKey(0)
############

contours_dict = []

# 6) contours 적용후 사각형으로 추출
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)

    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2), #중심좌표
        'cy': y + (h / 2)
    })
##############
#cv2.imshow('Image Basic',  temp_result)
#cv2.waitKey(0)
############

# 7) bound 사각형 정제1 (번호의 비율로 추출)
MIN_AREA = 80 #번호판 최소 너비
MIN_WIDTH, MIN_HEIGHT = 2, 8 # 2:8 비율
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h'] #넓이
    ratio = d['w'] / d['h'] #가로 세로 비율
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt +=1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)

##############
#cv2.imshow('Image Basic',  temp_result)
#cv2.waitKey(0)
############

# 8) bound 사각형 정제2 (번호끼리의 위치비율로 추출)
MAX_DIAG_MULTIPLYER = 5 # 번호판의 모든 번호가 bound의 5배 거리 내에 있어야함
MAX_ANGLE_DIFF = 12.0 # 첫번째 번호랑 마지막 번호의 기울기 세타 최대값
MAX_AREA_DIFF = 0.5 # 첫번째 번호랑 마지막 번호랑 면적차이
MAX_WIDTH_DIFF = 0.8 # 첫번째 번호랑 마지막 번호랑 각 번호의 너비차이
MAX_HEIGHT_DIFF = 0.2 # 첫번째 번호랑 마지막 번호랑 각 번호의 높이차이
MIN_N_MATCHED = 3 # 위로 인해 결과 걸러진것들이 연속으로 3개이상 되야 ok


def find_chars(contours_list):
    matched_result_idx = []
    for d1 in contours_list:
        matched_contours_idx = []
        for d2 in contours_list: #검증시 첫번째 번호판과 두번째 번호판 검증이 되면, 세번째, 네번째 번호판은 안해도됨
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            # 첫번째 번호판과 두번째 번호판의 거리 구하기
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            # 첫번쨰 번호판과 두번째 번호판의 각도 구하기
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx==0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) #면적비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w'] #너비비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h'] #높이비율

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])  #검증된 idx(2,3,4)만 넣음

        matched_contours_idx.append(d1['idx']) # 검증원천 idx 1번도 넣음

        if len(matched_contours_idx) < MIN_N_MATCHED: #후보군이 3개보다 적으면 탈락
            continue

        matched_result_idx.append(matched_contours_idx) #최종 후보군 넣음

        unmatched_contour_idx = []
        for d4 in contours_list: #탈락한 애들도 재 비교
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)


matched_result = []

for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)

##############
#cv2.imshow('Image Basic',  temp_result)
#cv2.waitKey(0)
############

# 9) 번호판을 가로 180도가 되도록 평편하게 변경 및 자르기
PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MMIN_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []


for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) /2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) /2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm( #첫번째 번호와 마지막번호의 거리
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) #번호판이 기울어진 각도

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height)) #180도가 되도록 회전시킴

    img_cropped = cv2.getRectSubPix( #이미지 자르기
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue
    
    plate_imgs.append(img_cropped)


##############
#cv2.imshow('Image Basic',  img_cropped)
#cv2.waitKey(0)
############



# 10) 한번더 스레쉬홀딩(문자 찾기위해)
longest_idx, longest_text = -1, 0
plate_chars = []

for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0,0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY |
    cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)

    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h


    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_Result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY |
    cv2.THRESH_OTSU)

    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0)) #패딩
    
    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0') #psm7 이미지 안에 한줄 가정. oem 0은 가장 초기모델(원시적. 문맥없음)

    
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit:
                has_digit = True
            result_chars += c
    
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

# 11) 사진과 번호 출력
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
         print(e)
         return False


print(result_chars)
imwrite(result_chars + '.jpg', img_ori)