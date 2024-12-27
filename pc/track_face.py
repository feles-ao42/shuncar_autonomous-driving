import cv2


class CameraHandler:
    def __init__(self, camera_id=2):
        self.camera_id = camera_id
        self.cap = None

    def start(self):
        """カメラを開始する"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")

    def stop(self):
        """カメラを停止する"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()

    def read_frame(self):
        """フレームを読み込む"""
        if self.cap is None:
            raise RuntimeError("カメラが開始されていません")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame):
        """顔を検出して位置とサイズを返す"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            # 位置の判定
            center_x = x + w // 2
            if center_x < frame.shape[1] // 3:
                position = "左"
            elif center_x > 2 * frame.shape[1] // 3:
                position = "右"
            else:
                position = "中央"

            # サイズの判定
            area = w * h
            if area < 20000:
                size = "小"
            elif area < 100000:
                size = "中"
            else:
                size = "大"

            results.append({
                'position': position,
                'size': size,
                'coords': (x, y, w, h)
            })

        return results

    def draw_results(self, frame, results):
        """検出結果を描画する"""
        for result in results:
            x, y, w, h = result['coords']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame,
                        f"{result['position']}, {result['size']}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2)
        return frame


def main():
    camera = CameraHandler(camera_id=2)
    detector = FaceDetector()

    try:
        camera.start()
        while True:
            frame = camera.read_frame()
            if frame is None:
                break

            results = detector.detect(frame)
            for result in results:
                print(f"位置: {result['position']}, サイズ: {result['size']}")

            frame = detector.draw_results(frame, results)

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()


if __name__ == "__main__":
    main()
