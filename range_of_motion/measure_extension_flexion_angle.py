import cv2
import numpy as np
import math

def findCentroid(thresh_img):
    moments = cv2.moments(thresh_img)

    area = moments['m00']

    if area > 0:
        x = int(moments['m10'] / area)
        y = int(moments['m01'] / area)
        return (x, y)
    return -1

def findAngle(pt1, pt2):
    angle1 = np.arctan2(*pt1[::-1])
    angle2 = np.arctan2(*pt2[::-1])
    angle = np.rad2deg((angle1 - angle2) % (2 * np.pi))
    # angle = int(math.atan((pt1[1]-pt2[1])/(pt2[0]-pt1[0]))*180/math.pi)
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

    frame_num = 0
    neutral_angles = []
    extension_angles = []
    flexion_angles = []

    while True:
        _, frame = cap.read()

        denoise_frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_frame = cv2.cvtColor(denoise_frame, cv2.COLOR_BGR2HSV)

        thresh_red = cv2.inRange(hsv_frame, (165, 100, 40), (255, 255, 200))
        thresh_blue = cv2.inRange(hsv_frame, (100, 100, 40), (120, 255, 200))

        red_centroid = findCentroid(thresh_red)
        blue_centroid = findCentroid(thresh_blue)

        if frame_num < 100:
            cv2.putText(frame, 'Maintain neutral angle', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        elif 100 < frame_num < 250:
            cv2.putText(frame, 'Extend wrist', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        elif 250 < frame_num < 400:
            cv2.putText(frame, 'Flex wrist', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        if red_centroid != -1 and blue_centroid != -1:
            cv2.circle(frame, red_centroid, 2, (0, 0, 255), 10)
            cv2.circle(frame, blue_centroid, 2, (255, 0, 0), 10)
            cv2.line(frame, red_centroid, blue_centroid, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.rectangle(frame, (20, 0), (130, 40), (0, 0, 0), thickness=-1, lineType=8, shift=0)
            cv2.putText(frame, '{:3.2f}'.format(findAngle(red_centroid, blue_centroid)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            if frame_num < 100:
                neutral_angles.append(findAngle(red_centroid, blue_centroid))
            elif 150 < frame_num < 250:
                extension_angles.append(findAngle(red_centroid, blue_centroid))
            elif 300 < frame_num < 400:
                flexion_angles.append(findAngle(red_centroid, blue_centroid))

        if frame_num == 400:
            neutral_angle = np.sum(neutral_angles) / len(neutral_angles)
            extension_angle = max(extension_angles, key=abs) - neutral_angle
            flexion_angle = 360 - abs(neutral_angle - max(flexion_angles, key=abs))

            print('Reference angle: {}'.format(neutral_angle))
            print('Extension angle: {}'.format(extension_angle))
            print('Flexion angle: {}'.format(flexion_angle))
            break

        # cv2.imshow('Red', thresh_red)
        # cv2.imshow('Blue', thresh_blue)
        cv2.imshow('Frame', frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_num += 1

    out_image = cv2.imread('./wrist_extension_flexion.jpg')
    cv2.putText(out_image, 'Max extension angle: {:3.2f} degrees'.format(extension_angle), (375, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))
    cv2.putText(out_image, 'Max flexion angle: {:3.2f} degrees'.format(flexion_angle), (340, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))
    cv2.imshow('Out', out_image)
    cv2.waitKey(0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
