import cv2
import numpy as np
from sklearn.cluster import KMeans

# Configuración general
camera_base_id = 0
camera_lateral_id = 1
camera_efector_id = 0

def procesar_camara_base():
    cap = cv2.VideoCapture(camera_base_id)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara en la base.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([165, 100, 100])    # actualmente es el color rosa // ns splatoon joy-con
        upper_hsv = np.array([175, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
        
        edges = cv2.Canny(mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print(f"Centroide del efector: ({cx}, {cy})")
        
        cv2.imshow("Cámara Base", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def procesar_camara_lateral():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)  # Ajustar si es necesario
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara lateral.")
        return

    # Inicializar el descriptor HOG para la detección de personas
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Rango de color para la detección de objetos (ajustar estos valores para el objeto específico)
    lower_color = np.array([165, 100, 100])  # Límite inferior en HSV
    upper_color = np.array([175, 255, 255])  # Límite superior en HSV

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el fotograma.")
                break

            # Convertir a HSV para la detección de color
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Crear una máscara para el objeto coloreado
            mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # Encontrar los contornos del objeto coloreado
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            # Variables para almacenar posiciones
            posicion_objeto = None
            posicion_persona = None

            # Detectar personas en el fotograma
            boxes, weights = hog.detectMultiScale(frame, 
                                                  winStride=(8, 8),
                                                  padding=(4, 4),
                                                  scale=1.05)

            # Encontrar la detección de persona más grande
            mayor_area_persona = 0
            for (x, y, w, h) in boxes:
                area = w * h
                if area > mayor_area_persona:
                    mayor_area_persona = area
                    # Usar el punto central inferior del cuadro delimitador
                    posicion_persona = (x + w//2, y + h)
                    # Dibujar un rectángulo alrededor de la persona
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Encontrar el objeto coloreado
            if contours:
                # Obtener el objeto coloreado más grande
                contorno_mas_grande = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contorno_mas_grande) > 100:  # Umbral mínimo de área
                    # Obtener el centro del objeto
                    M = cv2.moments(contorno_mas_grande)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        posicion_objeto = (cx, cy)
                        # Dibujar un círculo en el centro del objeto
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.drawContours(frame, [contorno_mas_grande], -1, (0, 0, 255), 2)

            # Calcular y mostrar la distancia si tanto el objeto como la persona son detectados
            if posicion_objeto and posicion_persona:
                # Calcular la distancia euclidiana en píxeles
                distancia = np.sqrt(
                    (posicion_objeto[0] - posicion_persona[0])**2 + 
                    (posicion_objeto[1] - posicion_persona[1])**2
                )
                
                # Dibujar una línea entre el objeto y la persona
                cv2.line(frame, posicion_objeto, posicion_persona, (255, 0, 0), 2)
                
                # Mostrar la distancia en el fotograma
                posicion_texto = (
                    (posicion_objeto[0] + posicion_persona[0]) // 2,
                    (posicion_objeto[1] + posicion_persona[1]) // 2
                )
                cv2.putText(frame, f"Distancia: {distancia:.1f}px", 
                            posicion_texto,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                print(f"Distancia al objeto: {distancia:.1f} píxeles")

            # Mostrar el fotograma
            cv2.imshow("Cámara Lateral", frame)
            
            # Salir del bucle con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def procesar_camara_efector():
    """
    Process feed from end effector camera to detect dominant colors
    and their spatial distribution in the image.
    """
    cap = cv2.VideoCapture(CAMERA_EFECTOR_ID)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara en el efector.")
        return

    # Define grid size for region analysis
    GRID_SIZE = (4, 4)  # Divides image into 4x4 grid

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el frame.")
                break

            # Get frame dimensions
            height, width = frame.shape[:2]
            cell_height = height // GRID_SIZE[0]
            cell_width = width // GRID_SIZE[1]

            # Create a copy for visualization
            display_frame = frame.copy()

            # Draw grid lines
            for i in range(1, GRID_SIZE[0]):
                cv2.line(display_frame, (0, i * cell_height),
                        (width, i * cell_height), (255, 255, 255), 1)
            for j in range(1, GRID_SIZE[1]):
                cv2.line(display_frame, (j * cell_width, 0),
                        (j * cell_width, height), (255, 255, 255), 1)

            # Analyze each grid cell
            for i in range(GRID_SIZE[0]):
                for j in range(GRID_SIZE[1]):
                    # Extract cell region
                    y_start = i * cell_height
                    y_end = (i + 1) * cell_height
                    x_start = j * cell_width
                    x_end = (j + 1) * cell_width
                    
                    cell = frame[y_start:y_end, x_start:x_end]
                    
                    # Reshape cell for k-means
                    cell_reshaped = cell.reshape((-1, 3)).astype(np.float32)
                    
                    # Perform k-means clustering on cell
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
                    kmeans.fit(cell_reshaped)
                    
                    # Get the two most dominant colors in the cell
                    labels_count = np.bincount(kmeans.labels_)
                    top_two_indices = np.argsort(-labels_count)[:2]
                    colors = kmeans.cluster_centers_[top_two_indices]
                    percentages = labels_count[top_two_indices] / len(kmeans.labels_)
                    
                    # Draw color information for this cell
                    text_y = y_start + 20
                    for idx, (color, percentage) in enumerate(zip(colors, percentages)):
                        # Convert color to BGR integers
                        color_bgr = color.astype(int).tolist()
                        
                        # Draw color sample
                        sample_x = x_start + 5
                        sample_y = text_y - 15 + (idx * 30)
                        cv2.rectangle(display_frame, 
                                    (sample_x, sample_y),
                                    (sample_x + 20, sample_y + 20),
                                    color_bgr, -1)
                        
                        # Add percentage text
                        text = f"{percentage*100:.1f}%"
                        cv2.putText(display_frame, text,
                                  (sample_x + 25, sample_y + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                  (255, 255, 255), 1)
                        
                        # Print region information to console
                        region_name = f"Region ({i+1},{j+1})"
                        print(f"{region_name} - Color {idx+1}: BGR{color_bgr}, {percentage*100:.1f}%")

            # Show the frame with grid and color information
            cv2.imshow("Cámara Efector - Análisis de Color", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Constants for camera IDs (assuming they're defined elsewhere)
CAMERA_EFECTOR_ID = 0

if __name__ == "__main__":
    opcion = input("Selecciona la cámara a procesar (base/lateral/efector): ")
    if opcion == "base":
        procesar_camara_base()
    elif opcion == "lateral":
        procesar_camara_lateral()
    elif opcion == "efector":
        procesar_camara_efector()
    else:
        print("Opción no válida.")