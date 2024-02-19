from ultralytics import YOLO
import cv2
from time import time
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
import supervision as sv

from timer import PlayerTimer

timer = PlayerTimer()

video_name = 'data/vid1.mp4'
cap = cv2.VideoCapture(video_name)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f'video fps are {fps}')

model = YOLO('models/best.pt')

inner_polygon = [1056, 326],[604, 374],[812, 490],[1272, 422]
# inner_polygon = Polygon(inner_polygon)
inner_pts = np.array(inner_polygon,
               np.int32)
inner_pts = inner_pts.reshape((-1, 1, 2))

outer_polygon = [864, 266],[360, 310],[564, 710],[1268, 574],[1272, 378]
# outer_polygon = Polygon(outer_polygon)
outer_pts = np.array(outer_polygon,
               np.int32)
 
outer_pts = outer_pts.reshape((-1, 1, 2))
isClosed = True
 
# Blue color in BGR
color = (255, 0, 0)
 
# Line thickness of 2 px
thickness = 4
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
txt_col = (0, 255, 255) 
  
# Line thickness of 2 px 
txt_th = 2

# class_names = {
#     0: 'ball',
#     1: 'basket',
#     2: 'referee',
#     3: 'player'
# }

if cap.isOpened():
    while True:
        t0 = time()
        ret, frame = cap.read()
        if ret:
            results = model.track(frame, imgsz=[1280, 736],
                                conf=0.1,    
                                verbose=False,
                                persist=True, 
                                tracker='botsort.yaml')
            frame = results[0].plot()
            detections = sv.Detections.from_ultralytics(results[0])
            # ball = detections[detections.class_id==0]
            # if ball.xyxy.size>0:
            #     ball = ball.xyxy[0]
            #     ball = box(*ball)
            #     if poly.intersects(ball):
            total_frames, is_basket = timer(detections) 
            if is_basket:
                total_time = {key: f'{str(round(value / fps, 2))} s' for key, value in total_frames.items()}
                print(total_time)
                timer.reset()
            frame = cv2.polylines(frame, [outer_pts], 
                      isClosed, color, thickness)
            frame = cv2.polylines(frame, [inner_pts], 
                      isClosed, color, thickness)
            if timer.pick_n_roll:
                frame = cv2.putText(frame, "Pick'n Roll", org, font,  
                    fontScale, txt_col, txt_th, cv2.LINE_AA) 
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # cv2.imwrite('image.png', frame)
                break
        else:
            cv2.destroyAllWindows()
            break
    total_time = {key: f'{str(round(value / fps, 2))} s' for key, value in total_frames.items()}
    print(total_time)
        # print("FPS: ", 1/(time()-t0))
else:
    print(f'No {video_name} file found!!!')
    