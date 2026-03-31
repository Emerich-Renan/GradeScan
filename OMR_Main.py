import cv2
import numpy as np
import utils

########################################
path  = "1.jpg"
widthImg = 1000
heightImg = 1000
########################################

img = cv2.imread(path)

# PRE-PROCESSING
img = cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)

imgThresh = cv2.adaptiveThreshold(
    imgBlur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

kernel = np.ones((3,3), np.uint8)
imgDil = cv2.dilate(imgThresh, kernel, iterations=1)

# CONTORNOS
countours, hierarchy = cv2.findContours(
    imgDil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------
# DETECTAR TODAS AS BOLINHAS
# -----------------------------
bubbleContours = []

for cnt in countours:
    area = cv2.contourArea(cnt)

    if 160 < area < 250:
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if circularity > 0.7:

            x, y, w, h = cv2.boundingRect(cnt)
            roi = imgThresh[y:y+h, x:x+w]

            filled = cv2.countNonZero(roi)
            total = w * h
            fillRatio = filled / float(total)

            # 🔥 pega vazias e preenchidas (ignora texto)
            if 0.3 < fillRatio < 0.9:
                bubbleContours.append(cnt)
                cv2.drawContours(imgContours, [cnt], -1, (0,255,0), 2)

print("Total bolinhas:", len(bubbleContours))

# -----------------------------
# SEPARAR COLUNAS
# -----------------------------
left = []
right = []

for cnt in bubbleContours:
    x, y, w, h = cv2.boundingRect(cnt)

    if x < widthImg // 2:
        left.append(cnt)
    else:
        right.append(cnt)

# ordenar por Y
left = sorted(left, key=lambda c: cv2.boundingRect(c)[1])
right = sorted(right, key=lambda c: cv2.boundingRect(c)[1])

bubbleContours = left + right

# -----------------------------
# AGRUPAR QUESTÕES (5 por linha)
# -----------------------------
questions = []
for i in range(0, len(bubbleContours), 5):
    questions.append(bubbleContours[i:i+5])

# -----------------------------
# DETECTAR RESPOSTAS
# -----------------------------
answers = []
alternatives = ["A", "B", "C", "D", "E"]

for q in questions:
    pixels = []

    # ordenar A → E
    q = sorted(q, key=lambda c: cv2.boundingRect(c)[0])

    for cnt in q:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = imgThresh[y:y+h, x:x+w]

        totalPixels = cv2.countNonZero(roi)
        pixels.append(totalPixels)

    print(pixels)  # DEBUG

    maxVal = max(pixels)
    minVal = min(pixels)

    # 🔥 decisão baseada na diferença
    if (maxVal - minVal) > 10:
        markedIndex = pixels.index(maxVal)
        answers.append(alternatives[markedIndex])
    else:
        answers.append("V")  # vazio ou dúvida

# -----------------------------
# RESULTADO
# -----------------------------
print("\nRespostas detectadas:")
print(answers)

# -----------------------------
# EXIBIÇÃO
# -----------------------------
imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray,imgBlur,imgThresh],
               [imgContours,imgBlank,imgBlank,imgBlank])
imgStacked = utils.stackImages(imageArray, 0.47)

cv2.imshow("Stacked Images",imgStacked)
cv2.waitKey(0)
cv2.destroyAllWindows()