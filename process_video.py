#process a video using our model
#firstly, video->frames
#secondly, transfer the style of these frames
#finally, frames->a video


import cv2
#video->frames
vc=cv2.VideoCapture('./video.mp4')
c=1
if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    rval,frame=vc.read()
    cv2.imwrite('./video_image/'+str(c)+'.jpg',frame)
    c=c+1
    cv2.waitKey(1)
vc.release()

#framess->video
import os
fps = 10
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
video_writer1 = cv2.VideoWriter(filename='./stylized_video.mp4', fourcc=fourcc, fps=fps, frameSize=(512,512))
for i in range(1,662):
  p = i
  if os.path.exists('./video_image/'+str(p)+'.jpg'):  
    img = cv2.imread(filename='./video_image/'+str(p)+'.jpg')
    cv2.waitKey(100)
    video_writer1.write(img)
    print(str(p) + '.jpg' + ' done!')
  else:
        print('no')
video_writer1.release()
