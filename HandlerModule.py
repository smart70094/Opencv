from abc import ABCMeta, abstractmethod
from DetectToolsModule import DetectTools
import cv2
import datetime
from AppConfigModule import AppConfig
import numpy as np


class HandlerMgr:
    __processors = []

    def add(self,handler):
        self.__processors.append(handler)

    def execute(self, problem_map):
        for p in self.__processors:
            p.resolve(problem_map)
        return problem_map['image_np']


class Handler(metaclass=ABCMeta):

    @abstractmethod
    def resolve(self,proble_map):
        pass


class DetectFaceHandler(Handler):

    def resolve(self,problem_map):
        image_np = problem_map['image_np']
        gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        rects = DetectTools.face_cascade.detectMultiScale(
                    gray_image_np,
                    scaleFactor= AppConfig.scaleFactor,
                    minNeighbors= AppConfig.minNeighbors,
                    minSize= AppConfig.minSize,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        problem_map['detect_faces_rects'] = rects
        rects_np = np.array(rects)
        if rects_np.size > 0 :
            problem_map['is_face'] = True
        else:
            problem_map['is_face'] = False


class DiffImageHandler(Handler):

    def resolve(self,problem_map):
        image_np = problem_map['image_np']
        base_image_np = problem_map['base_image_np']
        rects = DetectTools.differ(base_image_np,image_np)
        problem_map['detect_rects'] = rects


class TrackerFaceHandler(Handler):
    track_record = {}

    def resolve(self,problem_map):
        image_np = problem_map['image_np']
        detect_rects = problem_map['detect_rects']
        is_face = problem_map['is_face']

        # if has face in frame, then update record list.
        if is_face:
            for rect in detect_rects:
                x, y, w, h = rect
                rect_center_x = x + w / 2
                rect_center_y = y + h / 2
                matched_id = None

                for track_id in self.track_record.keys():
                    track_rect = self.track_record[track_id]['target_object']
                    t_x, t_y, t_w, t_h = track_rect
                    track_rect_center_x = t_x + t_w / 2
                    track_rect_center_y = t_y + t_h / 2

                    if ((t_x <= rect_center_x <= (t_x + t_w)) and
                            (t_y <= rect_center_y <= (t_y + t_h)) and
                            (x <= track_rect_center_x <= (x + w)) and
                            (y <= track_rect_center_y <= (y + h))):
                        matched_id = track_id
                        self.track_record[track_id]['target_object'] = rect
                        break

                if matched_id is None:
                    new_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    # add image rect
                    self.track_record[new_id] = {}
                    self.track_record[new_id]['target_object'] = rect
                    # add tracker
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(image_np, tuple(rect))
                    self.track_record[new_id]['tracker'] = tracker
                    print("new track:%s" % new_id)

        problem_map['track_faces_rects'] = []
        prepare_delete_record = []
        for key in self.track_record.keys():
            tracker = self.track_record[key]['tracker']

            ret,track_faces_rects = tracker.update(image_np)

            if ret:
                problem_map['track_faces_rects'].append(track_faces_rects)
            else:
                print("track failure")
                prepare_delete_record.append(key)

        for key in prepare_delete_record:
            self.track_record.pop(key, None)


class MarkFaceHandler(Handler):

    def resolve(self, problem_map):
        image_np = problem_map['image_np']
        detect_rects = problem_map.get('detect_rects',None)
        if detect_rects is not None:
            for rect in detect_rects:
                DetectTools.mark(image_np, rect, (0, 255, 0))

        track_faces_rects = problem_map.get('track_faces_rects',None)
        if track_faces_rects is not None:
            for rect in track_faces_rects:
                DetectTools.mark(image_np, rect, (0, 0, 255))

        detect_face_rects = problem_map.get('detect_faces_rects', None)
        if detect_face_rects is not None:
            for rect in detect_face_rects:
                DetectTools.mark(image_np, rect, (255, 0, 0))
