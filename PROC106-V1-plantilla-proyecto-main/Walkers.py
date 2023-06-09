import cv2

# Cargar el clasificador de cuerpos pre-entrenado
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Inicializar la captura de video para nuestro archivo de video
cap = cv2.VideoCapture('walking.avi')

# Comenzar el bucle una vez que el video esté cargado exitosamente
while True:
    # Leer el primer cuadro
    ret, frame = cap.read()

    if not ret:
        break

    # Convertir cada cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasar el cuadro a nuestro clasificador de cuerpos
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

    # Extraer las cajas envolventes para cualquier cuerpo identificado
    for (x, y, w, h) in bodies:
        # Dibujar un rectángulo alrededor del cuerpo detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Mostrar el cuadro con las cajas envolventes dibujadas
    cv2.imshow('Body Detection', frame)

    if cv2.waitKey(1) == 32:  # 32 es la barra espaciadora
        break

cap.release()
cv2.destroyAllWindows()
