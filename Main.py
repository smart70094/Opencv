from WidgetModule import MainWidget, Thread
from PyQt5.QtWidgets import QApplication
import sys

from HandlerModule import *
import time


def main2():
    app = QApplication(sys.argv)
    th = Thread()
    mainWidget = MainWidget(app, th)
    sys.exit(app.exec_())


def main():
    # define process
    handler_mgr = HandlerMgr()
    handler_mgr.add(DiffImageHandler())
    handler_mgr.add(DetectFaceHandler())
    handler_mgr.add(TrackerFaceHandler())
    handler_mgr.add(MarkFaceHandler())

    # initial background
    camera = cv2.VideoCapture(1)
    base_image_np = DetectTools.init(camera)
    time.sleep(3)

    # run main part
    problem_map = {}
    problem_map['base_image_np'] = base_image_np
    while True:
        ret, image_np = camera.read()
        if ret:
            problem_map['image_np'] = image_np
            image_np = handler_mgr.execute(problem_map)
        cv2.imshow('windwos', image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
