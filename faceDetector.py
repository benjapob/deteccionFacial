import cv2

# Carga una data entrenada en caras frontales de opencv (haar cascade algorithm)
trainedFaceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Elige una foto para detectar la cara
img = cv2.imread("brendan.webp")

# Capturar video de una camara
# webcam = cv2.VideoCapture("")

# Iterar los frames infinitamente
while True:
    # Ver el primer frame
    # successful_frame_read, frame = webcam.read()

    # Imagen convertida a escala de grises
    grayscaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faceCoordinates = trainedFaceData.detectMultiScale(grayscaledImg)

    # Dibujar rectangulos en las caras

    for x, y, w, h in faceCoordinates:
        cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 2)

    # Mostrar la imagen con los rectangulos
    cv2.imshow("Face detector", img)

    # Esperar hasta cerrar la ventana
    key = cv2.waitKey(1)

    # Parar si la letra Q es presionada
    if key == 81 or key == 113:
        break

# Limpiar el objeto de captura de video
img.release()


# print(faceCoordinates[0])
# (x, y, w, h) = faceCoordinates[0]
# cv2.rectangle(img, (x,y,w,h), (0, 255, 0), 2)

# print(faceCoordinates)

"""# Mostrar la imagen con los rectangulos
cv2.imshow('Face detector', img)

# Esperar hasta cerrar la ventana
cv2.waitKey()"""
