import cv2
import numpy as np
import os
import fitz  # PyMuPDF

########################################
path_gabarito = "gabarito.jpg"  # gabarito
pasta_respostas = "respostas"   # pasta com PDFs/JPGs dos cartões
widthImg = 1000
heightImg = 1000
########################################

# -----------------------------
# FUNÇÃO DE PROCESSAMENTO
# -----------------------------
def processar(img):
    img = cv2.resize(img,(widthImg,heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgThresh = cv2.adaptiveThreshold(
        imgBlur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )
    kernel = np.ones((3,3), np.uint8)
    imgDil = cv2.dilate(imgThresh, kernel, iterations=1)
    cv2.imshow("Thresh", imgThresh)
    cv2.waitKey(0)

    countours, _ = cv2.findContours(imgDil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bubbleContours = []

    for cnt in countours:
        area = cv2.contourArea(cnt)
        if 150 < area < 220:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.7:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = imgThresh[y:y+h, x:x+w]
                filled = cv2.countNonZero(roi)
                total = w * h
                fillRatio = filled / float(total)
                if 0.3 < fillRatio < 0.9:
                    bubbleContours.append(cnt)

    # separar esquerda/direita
    left, right = [], []
    for cnt in bubbleContours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < widthImg // 2: left.append(cnt)
        else: right.append(cnt)
    left = sorted(left, key=lambda c: cv2.boundingRect(c)[1])
    right = sorted(right, key=lambda c: cv2.boundingRect(c)[1])
    bubbleContours = left + right

    # separar por questão
    questions = []
    for i in range(0, len(bubbleContours), 5):
        questions.append(bubbleContours[i:i+5])

    alternatives = ["A","B","C","D","E"]
    answers = []

    for q in questions:
        pixels = []
        q = sorted(q, key=lambda c: cv2.boundingRect(c)[0])
        for cnt in q:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = imgThresh[y:y+h, x:x+w]
            pixels.append(cv2.countNonZero(roi))
        if len(pixels)==5:
            maxVal = max(pixels)
            minVal = min(pixels)
            if (maxVal - minVal) > 10:
                markedIndex = pixels.index(maxVal)
                answers.append(alternatives[markedIndex])
            else:
                answers.append("V")
    return answers

# -----------------------------
# FUNÇÃO DE CORREÇÃO
# -----------------------------
def corrigir(respostas, gabarito):
    acertos = 0
    erros = []
    for i in range(len(gabarito)):
        if i < len(respostas) and respostas[i]==gabarito[i]:
            acertos += 1
        else:
            erros.append(i+1)
    return acertos, erros

# -----------------------------
# CARREGAR GABARITO
# -----------------------------
if not os.path.exists(path_gabarito):
    print(f"Gabarito não encontrado em {path_gabarito}")
    exit()
imgGabarito = cv2.imread(path_gabarito)
gabarito = processar(imgGabarito)
print("\nGABARITO:")
print(gabarito)

# -----------------------------
# PROCESSAR ARQUIVOS
# -----------------------------
if not os.path.exists(pasta_respostas):
    print(f"Pasta {pasta_respostas} não existe!")
    exit()

arquivos = os.listdir(pasta_respostas)
print("\nCORREÇÃO EM MASSA:")

for arquivo in arquivos:
    caminho = os.path.join(pasta_respostas, arquivo)

    # IMAGENS
    if arquivo.lower().endswith((".jpg",".jpeg",".png")):
        img = cv2.imread(caminho)
        if img is None:
            print(f"{arquivo}: erro ao carregar")
            continue
        respostas = processar(img)
        acertos, erros = corrigir(respostas, gabarito)
        print(f"{arquivo}: {acertos}/{len(gabarito)} | erros: {erros}")

    # PDF
    elif arquivo.lower().endswith(".pdf"):
        try:
            doc = fitz.open(caminho)
        except Exception as e:
            print(f"{arquivo}: erro ao abrir PDF ({e})")
            continue

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=400)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n==4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            respostas = processar(img)
            acertos, erros = corrigir(respostas, gabarito)
            print(f"{arquivo} pág {i+1}: {acertos}/{len(gabarito)} | erros: {erros}")
