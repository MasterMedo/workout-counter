import cv2 as cv
import numpy as np

from repetition_detector import f2

REZOLUCIJA_SLANJA_F2 = 5    # Svakih koliko se šalje u F2
cap = cv.VideoCapture(0)
ret, prvi_okvir = cap.read()
prosli_sivi = cv.cvtColor(prvi_okvir, cv.COLOR_BGR2GRAY)
maska = np.zeros_like(prvi_okvir)
maska[..., 1] = 255

# Alternator
prvi_ili_drugi = 1

# Par za Branimira
flow_grlice = []

frame_num_counter = 0
# ZG petlja
while (cap.isOpened()):
    frame_num_counter += 1
    validno, okvir = cap.read()
    cv.imshow("Kamera", okvir)

    if not validno:
        break
    sivi = cv.cvtColor(okvir, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prosli_sivi, sivi,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    # Za F2
    if frame_num_counter == 1:
        flow_grlice.append(flow)
    else:
        if frame_num_counter % REZOLUCIJA_SLANJA_F2 == 0:
            if len(flow_grlice) == 1:
                flow_grlice.append(flow)
            else:
                flow_grlice[0] = flow_grlice.pop(1)
                flow_grlice.append(flow)

            # Šalje se par flowova - zovi F2
            print(len(flow_grlice))
            f2(flow_grlice[0], flow_grlice[1])
    # Flowchart
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    maska[..., 0] = angle * 180 / np.pi / 2
    maska[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(maska, cv.COLOR_HSV2BGR)
    cv.imshow("DOF", rgb)

    # Kraj
    prosli_sivi = sivi
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
