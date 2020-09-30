from cv2 import cv2

def main_Detection():
    cap = cv2.VideoCapture('The Flash.mp4')
    while cap.isOpened():
        ret,frame = cap.read()
        frame = cv2.resize(frame,None, fx=1.1,fy=1.1,interpolation=cv2.INTER_AREA)
        if ret == True:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier('data_cascades/haarcascade_frontalface_alt2.xml')
            part_rects = cascade.detectMultiScale(gray,scaleFactor=1.65,minNeighbors=5)
            for (x,y,w,h) in part_rects:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

            cv2.imshow('video_face_detection',frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	main_Detection()