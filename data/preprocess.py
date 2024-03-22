import os
import cv2
import numpy as np
from tqdm import tqdm


class ImageData():
    def __init__(self, line_data, image_dir, image_size):
        self.image_size = image_size
        data = line_data.strip().split() ## 0~195 : landmark, 196~199 : bbox

        self.landmark = np.array(list(map(float, data[:196])), dtype=np.float32).reshape(-1, 2)
        self.bbox = np.array(list(map(int, data[196 : 200])), dtype=np.int32)
        
        flags = tuple(map(int, data[200:206]))
        self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur = flags

        self.image_file_path = f"{image_dir}/{data[206]}"
        self.images, self.landmarks, self.boxes = [], [], []

    def load_data(self, is_train, repeat, mirror=None):
        if mirror is not None:
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))

        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1
        center = (xy + wh/2).astype(np.int32)

        img = cv2.imread(self.image_file_path)
        boxsize = int(np.max(wh) * 1.2)
        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark+0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy)/boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.images.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.images) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))

                
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:,0] = 1 - landmark[:,0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.images.append(imgT)
                self.landmarks.append(landmark)


    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))

        labels = []
        bbox_labels = []
        ## tracked points를 기준으로 yaw, pitch, roll을 정량적으로 계산한다.
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.images, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            bbox_str = ' '.join(list(map(str, self.bbox.tolist())))
            bbox_label = f"{save_path} {bbox_str}\n"
            bbox_labels.append(bbox_label)

            euler_angles_landmark = []
            for index in TRACKED_POINTS: ## tracked_points에 해당하는 landmark(x, y)를 append
                euler_angles_landmark.append(lanmark[index])

            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28)) ## (1, 2 * 14)
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0]) ## (28,)
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))
            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = f"{save_path} {landmark_str} {attributes_str} {euler_angles_str}\n"
            labels.append(label)

        return labels, bbox_labels
    

def calculate_pitch_yaw_roll(landmarks_2D, cam_w=256, cam_h=256, radians=False):
    assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    ## camera center x, y
    c_x = cam_w / 2
    c_y = cam_h / 2

    ## focus distance x, y
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x

    ## project 2d keypoints into 3d space
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0]) ## No distortion

    """
    실제 3D 얼굴 모델의 랜드마크 포인트를 정의
    - dlib (68 landmark) trached points
        TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    - wflw(98 landmark) trached points
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    """
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
        [-1.330353, 7.122144, 6.903745], # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142], # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414], # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654], # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652], # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652], # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531], # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],# MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667], # LOWER_LIP,
        [0.000000, -7.415691, 4.070434], # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    ## 주어진 3D 점들과 그에 대응하는 2D 이미지 점들을 이용하여 카메라(뷰어)의 위치와 방향(회전)을 찾는다.
    ## solvePnP를 사용하여 3D-2D 점 대응을 통해 회전 벡터(rvec)와 이동 벡터(tvec) 계산
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix, camera_distortion)

    rmat, _ = cv2.Rodrigues(rvec) ## 회전 벡터를 3x3 회전 행렬로 변환
    pose_mat = cv2.hconcat((rmat, tvec)) ## 회전 행렬과 이동 벡터를 결합하여 포즈 행렬을 생성
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat) ## 포즈 행렬을 분해하여 오일러 각도(피치, 요, 롤)를 포함한 여러 값을 반환
    return map(lambda k: k[0], euler_angles)  # euler_angles contain (pitch, yaw, roll)


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2], M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_


def get_datasets(save_dir, landmark_file, image_size=112, is_train=True):
    with open(landmark_file, "r") as f:
        save_img_dir = f"{save_dir}/imgs"
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)

        labels = []
        bbox_labels = []
        lines = f.readlines()
        for idx, line in enumerate(tqdm(lines)):
            data_obj = ImageData(line, IMG_DIR, image_size)
            image_file = data_obj.image_file_path
            data_obj.load_data(is_train, 10, MIRROR)

            _, filename = os.path.split(image_file)
            filename, _ = os.path.splitext(filename)

            label_txt, bbox_txt = data_obj.save_data(save_img_dir, str(idx) + "_" + filename)
            labels.append(label_txt)
            bbox_labels.append(bbox_txt)

            # if ((idx + 1) % 100) == 0:
            #     print('file: {}/{}'.format(idx + 1, len(lines)))

    with open(os.path.join(save_dir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)

    with open(os.path.join(save_dir, 'bboxes.txt'),'w') as f:
        for label in bbox_labels:
            f.writelines(label)
        

if __name__ == "__main__":
    IMG_SIZE = 112
    DATA_DIR = "/home/pervinco/Datasets/WFLW"
    IMG_DIR = f"{DATA_DIR}/WFLW_images"
    MIRROR = f"{DATA_DIR}/WFLW_annotations/Mirror98.txt"
    LANDMARK_FILES = [f"{DATA_DIR}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt",
                      f"{DATA_DIR}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"]
    
    output_dir = f"{DATA_DIR}/dataset"
    for landmark_file in LANDMARK_FILES:
        set_name = landmark_file.split("/")[-1].split(".")[0].split("_")[-1]
        save_dir = f"{output_dir}/{set_name}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        is_train = True
        if set_name == "test":
            is_train = False

        images = get_datasets(save_dir, landmark_file, IMG_SIZE, is_train)
        