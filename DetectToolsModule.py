import cv2
import datetime
import os
from skimage.measure import compare_ssim
import imutils
from AppConfigModule import AppConfig


class DetectTools:
    face_cascade = cv2.CascadeClassifier(AppConfig.face_haarcascade_path)
    # tracker = cv2.MultiTracker_create()
    # eye_cascade = cv2.CascadeClassifier(AppConfig.eye_haarcascade_path)

    @staticmethod
    def mark(image_np, rect,color_tuple):
        x = int(rect[0])
        y = int(rect[1])
        w = int(rect[2])
        h = int(rect[3])
        cv2.rectangle(image_np, (x, y), (x + w, y + h), color_tuple, 2)

    @staticmethod
    def differ(imageA, imageB):
        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM: {}".format(score))

        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        width = AppConfig.limit_similarity_width
        height = AppConfig.limit_similarity_height

        result = []
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            if w > width and h > height and score < AppConfig.compare_similarity_threshold:
                result.append((x, y, w, h))
        return result

    @staticmethod
    def compare(image_np_1, image_np_2):
        hist1 = cv2.calcHist([image_np_1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([image_np_2], [0], None, [256], [0, 256])
        result = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return result >= AppConfig.compare_similarity_threshold

    @staticmethod
    def crop(image_np,rect):
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        return image_np[y : y + h, x : x + w]

    @staticmethod
    def save(image_np,path,image_format):
        time_format = '%Y%m%d%H%M%S'
        filename = datetime.datetime.now().strftime(time_format) + "." + image_format
        cv2.imwrite(os.path.join(path, filename), image_np)
        print("Save:%s" % filename)

    @staticmethod
    def init(camera):
        while True:
            ret, frame = camera.read()
            if ret:
                cv2.imwrite("bg.jpg", frame)
                break
        return cv2.imread("bg.jpg")


def main():
    camera = cv2.VideoCapture(1)

    while True:
        ret, frame = camera.read()

        if ret:
            cv2.imshow("windows", frame)
            DetectTools.save(frame, r'C:\Users\Jim\PycharmProjects\DetectFace3.0\Image',"jpg")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
     main()
