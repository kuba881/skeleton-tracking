import cv2
import mediapipe as mp
import os

class SkeletonTracker:
    def __init__(self, video_path, target, output_fps=1):
        self.output_fps = output_fps
        self.target = target

        folders = os.listdir()
        if 'corr' not in folders:
            os.mkdir(os.path.join('','corr'))
        elif 'wrg' not in folders:
            os.mkdir(os.path.join('','wrg'))

        if target == 0:
            folder = 'wrong'
            f_h = 'wrg'
        elif target == 1:
            folder = 'correct'
            f_h = 'corr'
        else:
            print('Unknown target!')

        self.video_path = os.path.join(folder,video_path)
        self.output_file = f'{os.path.join(f_h,video_path[0:len(video_path)-4])}_joints.txt'

        # Initialize MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def write_to_file(self, frame_number, landmarks):
        with open(self.output_file, "a") as file:
            file.write(f"Frame {frame_number}:\n")
            for landmark in landmarks:
                file.write(f"{landmark.x}, {landmark.y}, {landmark.z}\n")
            file.write("\n")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_number = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % int(fps / self.output_fps) == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    self.write_to_file(frame_number, landmarks)

            frame_number += 1

        cap.release()

    def visualize_video(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                self.draw_landmarks(frame, landmarks)

            cv2.imshow("Skeleton", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_landmarks(self, frame, landmarks):
        for landmark in landmarks:
            height, width, _ = frame.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
