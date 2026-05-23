import cv2
import threading
import time
import json
import base64
from collections import Counter
from ultralytics import YOLO
from ollama import chat, ResponseError
from alert_manager import AudioAlertManager
from benchmark_logger import BenchmarkLogger

# Configuración Base
MODEL_YOLO = "yolov8s.pt" 
URL_VIDEO = "http://10.207.29.128:8080/video"
VLM_MODEL = "qwen3-vl:2b" # Tu modelo Visual Generativo
UMBRAL_CONFIANZA = 0.45

# Inicializaciones
print("\n[INIT] Cargando modelos YOLO y la API Local de Ollama...")
model = YOLO(MODEL_YOLO)
audio_manager = AudioAlertManager()
benchmark = BenchmarkLogger(filename="metricas_vision_ieee.csv")

# Variables Globales
running = True
ultimo_frame = None
alerta_actual = "Inicializando..."
ultimo_resumen = ""
analizando_ahora = False

def capturar_camara_sin_latencia():
    """Vacía el buffer de red constantemente (Latencia nula)."""
    global ultimo_frame, running
    cap = cv2.VideoCapture(URL_VIDEO)
    while running:
        ret, frame = cap.read()
        if ret:
            ultimo_frame = frame
        else:
            time.sleep(0.01)
    cap.release()

def analizar_escena_ollama(imagen_b64, contexto_yolo):
    """Hilo asíncrono puro: Infiere Qwen VLM con imagen + contexto de YOLO."""
    global analizando_ahora, alerta_actual
    
    analizando_ahora = True
    prompt = f"Eres un bastón inteligente. YOLO ha detectado: {contexto_yolo}. Basado en esto y tu observación de la imagen visual adjunta, lanza una breve advertencia si hay obstáculo inminente de chocar. Sé conciso."
    
    try:
        response = chat(
            model=VLM_MODEL,
            format='json',
            messages=[{
                'role': 'user',
                'content': prompt + "\nResponde EXCLUSIVAMENTE en formato JSON estricto: {\"peligro\": \"Alto/Medio/Bajo\", \"alerta_voz\": \"mensaje corto\"}",
                'images': [imagen_b64]
            }]
        )
        
        json_response = response.message.content
        print(f"\n[QWEN VLM] {json_response}")
        
        try:
            data = json.loads(json_response)
            alerta_texto = data.get("alerta_voz", "Bip")
            alerta_actual = alerta_texto
            
            # Solo lanzar TTS si no está despejado para no fatigar auditivamente
            if "despejad" not in alerta_texto.lower():
                audio_manager.enviar_alerta(alerta_texto)
                
        except json.JSONDecodeError:
            alerta_actual = "Error parseando JSON del VLM"
            
    except ResponseError as e:
        alerta_actual = f"Error Ollama HTTP"
        print(f"\n[ERROR OLLAMA] {e.error}")
    except Exception as e:
        alerta_actual = "Timeout / Falla VLM local."
        print(f"\n[ERROR VLM] {e}")
        
    analizando_ahora = False

print("\n🚀 Iniciando Pipeline de Visión Local y Telemetría IEEE...")
hilo_camara = threading.Thread(target=capturar_camara_sin_latencia, daemon=True)
hilo_camara.start()

# Opcional: Esperar a tener señal de cámara
while ultimo_frame is None and running:
    time.sleep(0.1)

cv2.namedWindow("Deteccion", cv2.WINDOW_NORMAL)
ultimo_tiempo_ollama = time.time()
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0

# Trackers de estado para el Audio Reactivo
ultimo_estado_anunciado = None
tiempo_ultima_alerta = 0
tiempo_alarma_persona = 0

while running:
    if ultimo_frame is None:
        continue

    frame = ultimo_frame.copy()
    tiempo_inicio_yolo = time.time()

    # Phase 1: Inferencia Local YOLO (Rápida)
    results = model(frame, verbose=False, conf=UMBRAL_CONFIANZA)
    annotated = results[0].plot()

    inference_ms = (time.time() - tiempo_inicio_yolo) * 1000

    # Phase 2: Pseudo-profundimetría
    objetos_info = []
    nivel_peligro = "Bajo"
    
    for box in results[0].boxes:
        clase = model.names[int(box.cls)]
        xywh = box.xywh[0].cpu().numpy()
        w, h = xywh[2], xywh[3]
        
        tamano_relativo = (w * h) / (frame.shape[0] * frame.shape[1])
        
        if tamano_relativo > 0.15:      # > 15% del frame → CRITICO
            distancia = "CRITICO"
            nivel_peligro = "Alto"
        elif tamano_relativo > 0.04:   # > 4% del frame → Cerca
            distancia = "Cerca"
            if nivel_peligro != "Alto": nivel_peligro = "Medio"
        else:
            distancia = "Lejos"

        objetos_info.append(f"1 {clase} {distancia}")

    conteo = Counter(objetos_info)
    ultimo_resumen = ", ".join([f"{k}" for k, v in conteo.items()]) if conteo else "camino totalmente despejado"

    json_pipeline_output = {
        "riesgo": nivel_peligro,
        "escenario": ultimo_resumen,
        "objetos": len(objetos_info),
        "latencia_yolo_ms": round(inference_ms, 1)
    }
    
    print(f"[JSON Stream] -> {json.dumps(json_pipeline_output)}")

    # ========================================================
    # GESTIÓN INMEDIATA DE ALERTAS DE AUDIO (YOLO)
    # ========================================================
    if nivel_peligro == "Alto" and (time.time() - tiempo_ultima_alerta > 3.0):
        audio_manager.enviar_alerta("Atención, riesgo alto detectado.")
        tiempo_ultima_alerta = time.time()
        ultimo_estado_anunciado = "Alto"
        
    elif nivel_peligro == "Medio" and ultimo_estado_anunciado != "Medio" and (time.time() - tiempo_ultima_alerta > 4.0):
        audio_manager.enviar_alerta("Precaución, objeto cercano.")
        tiempo_ultima_alerta = time.time()
        ultimo_estado_anunciado = "Medio"

    elif nivel_peligro == "Bajo" and ultimo_estado_anunciado != "Bajo":
        audio_manager.enviar_alerta("Camino despejado.")
        ultimo_estado_anunciado = "Bajo"

    # ========================================================
    # ALARMA ESPECÍFICA PARA PERSONAS (SOLO ALTO RIESGO)
    # ========================================================
    persona_en_peligro = False
    for obj in objetos_info:
        if "person" in obj.lower() and "CRITICO" in obj:
            persona_en_peligro = True
            break
            
    # Reducimos drásticamente el bloqueador (1.5s) para que la voz repita la emergencia
    # continuamente mientras la persona siga enfrente bloqueando.
    if persona_en_peligro and (time.time() - tiempo_alarma_persona > 1.5):
        alerta_msg = "Persona detectada, Riesgo Crítico."
        print(f"\n[!!! ALARMA YOLO !!!] -> VOZ: {alerta_msg.upper()}")
        audio_manager.enviar_alerta(alerta_msg)
        tiempo_alarma_persona = time.time()

    # Calculadora de FPS de Muestreo YOLO
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps_display = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()

    benchmark.record_frame(fps_display, inference_ms, len(objetos_info), nivel_peligro, ultimo_resumen)

    if time.time() - ultimo_tiempo_ollama > 4.0 and not analizando_ahora:
        frame_pequeno = cv2.resize(annotated, (480, 320))
        _, buffer = cv2.imencode('.jpg', frame_pequeno, [cv2.IMWRITE_JPEG_QUALITY, 55])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        threading.Thread(target=analizar_escena_ollama, args=(img_b64, ultimo_resumen), daemon=True).start()
        ultimo_tiempo_ollama = time.time()

    # Dibujado de HUD (Colores oscuros para máximo contraste)
    cv2.putText(annotated, f"VLM Alerta: {alerta_actual[:30]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Negro
    cv2.putText(annotated, f"Latencia YOLO: {int(inference_ms)}ms  |  {fps_display} FPS", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2) # Azul oscuro
    
    # Rojo oscuro, Naranja oscuro, Verde oscuro
    color_nivel = (0, 0, 139) if nivel_peligro == "Alto" else ((0, 100, 200) if nivel_peligro == "Medio" else (0, 100, 0))
    cv2.putText(annotated, f"Riesgo Visual: {nivel_peligro}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_nivel, 2)

    cv2.imshow("Deteccion", annotated)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Deteccion", cv2.WND_PROP_VISIBLE) < 1:
        running = False
        break

# Limpieza y generación de reporte
cv2.destroyAllWindows()
benchmark.export_csv()
print("\n Pipeline finalizado. Métrica CSV IEEE grabada.")

