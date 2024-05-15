import pathlib
import cv2

cascade_path = pathlib.Path("haarcascode_frontalface_default.xml")
print(cascade_path)

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture("FACEE.mp4")

while True:
      _, frame = camera.read()
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      face = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
      )

      for(x,y,width,height) in face:
            cv2.rectangle(frame,(x,y),(x+width,y+height),(255,255,0),2)
      cv2.imshow("Faces", frame)
      if cv2.waitkey(1) == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()