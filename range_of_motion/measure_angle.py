import cv2
import numpy as np

def findCentroid(thresh_img):
    moments = cv2.moments(thresh_img)

    area = moments['m00']

    if area > 0:
        x = int(moments['m10'] / area)
        y = int(moments['m01'] / area)
        return (x, y)
    return (0, 0)

def findAngle(pt1, pt2):
    angle1 = np.arctan2(*pt1[::-1])
    angle2 = np.arctan2(*pt2[::-1])
    angle = np.rad2deg((angle1 - angle2) % (2 * np.pi))
    return angle

def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(width), int(height)))

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        _, frame = cap.read()

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        thresh_red1 = cv2.inRange(hsv_frame, (165, 145, 100), (250, 210, 160))
        thresh_red2 = cv2.inRange(hsv_frame, (0, 145, 100), (10, 210, 160))
        thresh_red = cv2.add(thresh_red1, thresh_red2)

        thresh_blue = cv2.inRange(hsv_frame, (105, 180, 40), (120, 260, 100))

        red_centroid = findCentroid(thresh_red)
        blue_centroid = findCentroid(thresh_blue)

        if red_centroid != (0, 0) and blue_centroid != (0, 0):
            cv2.circle(frame, red_centroid, 2, (0, 0, 255), 10)
            cv2.circle(frame, blue_centroid, 2, (255, 0, 0), 10)
            cv2.line(frame, red_centroid, blue_centroid, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.rectangle(frame, (20, 0), (130, 40), (0, 0, 0), thickness=-1, lineType=8, shift=0)
            cv2.putText(frame, '{:3.2f}'.format(findAngle(red_centroid, blue_centroid)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        cv2.imshow('Frame', frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
