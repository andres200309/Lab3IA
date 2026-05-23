# 🦯 Bastón Inteligente con Visión por Computadora
### Sistema de Detección de Obstáculos en Tiempo Real — Lab 3

> **Pipeline Edge AI:** YOLOv8 + Qwen3-VL (VLM) + TTS Reactivo + Telemetría IEEE  
> **Plataforma:** Windows 10/11 | Python 3.12 | CPU/GPU

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Estructura de Archivos](#3-estructura-de-archivos)
4. [Requisitos Previos](#4-requisitos-previos)
5. [Instalación Paso a Paso](#5-instalación-paso-a-paso)
6. [Configuración de la Cámara](#6-configuración-de-la-cámara)
7. [Cómo Ejecutar](#7-cómo-ejecutar)
8. [Módulos del Sistema](#8-módulos-del-sistema)
9. [Lógica de Detección y Alertas](#9-lógica-de-detección-y-alertas)
10. [Métricas y Telemetría IEEE](#10-métricas-y-telemetría-ieee)
11. [Solución de Problemas](#11-solución-de-problemas)

---

## 1. Descripción del Proyecto

Este proyecto implementa un **sistema asistivo de visión artificial** diseñado para personas con discapacidad visual. Emula el concepto de un "bastón inteligente" que analiza el entorno en tiempo real mediante una cámara IP (celular Android) y emite alertas de voz habladas cuando detecta obstáculos.

### Características principales

| Feature | Detalle |
|---|---|
| Detección de objetos | YOLOv8s — 80 clases del dataset COCO |
| Comprensión semántica | Qwen3-VL 2B (Visual Language Model) vía Ollama |
| Alertas de voz | PowerShell SAPI5 (TTS nativo Windows) |
| Cámara | IP Webcam via HTTP/MJPEG (celular Android) |
| Métricas | CSV IEEE con FPS, latencia, RAM y nivel de peligro |

### ¿Por qué dos modelos?

El sistema combina dos enfoques complementarios:

- **YOLO** es extremadamente rápido (~100–150ms) pero solo devuelve etiquetas y coordenadas. No puede razonar sobre el contexto.
- **Qwen3-VL** puede razonar visualmente ("la silla está a tu derecha y el pasillo está despejado") pero es lento (2–8s). 

La solución: YOLO gestiona las alertas inmediatas en tiempo real, mientras Qwen3-VL analiza la escena en segundo plano cada 4 segundos para proveer contexto semántico adicional.

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CÁMARA IP (Celular)                       │
│              http://192.168.1.X:8080/video                   │
└────────────────────────┬────────────────────────────────────┘
                         │ MJPEG Stream
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         HILO 1: capturar_camara_sin_latencia()               │
│  Lee frames continuamente, descarta buffer para 0 latencia   │
│                  → ultimo_frame (global)                     │
└────────────────────────┬────────────────────────────────────┘
                         │ Frame fresco
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              HILO PRINCIPAL: Bucle de Procesado              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  FASE 1 — YOLOv8s (rápido: ~100-150ms)              │    │
│  │  • Detecta objetos y dibuja bounding boxes           │    │
│  │  • Calcula tamaño relativo de cada caja              │    │
│  │  • Asigna distancia: CRITICO / Cerca / Lejos         │    │
│  │  • Determina nivel_peligro: Alto / Medio / Bajo      │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                           │                                  │
│  ┌───────────────────────▼─────────────────────────────┐    │
│  │  FASE 2 — Sistema de Alertas (inmediatas)            │    │
│  │  • Alto  → voz c/3s: "Atención, riesgo alto"         │    │
│  │  • Medio → voz c/4s: "Precaución, objeto cercano"    │    │
│  │  • Bajo  → voz 1x:   "Camino despejado"              │    │
│  │  • Persona CRITICO → voz c/1.5s: "Riesgo Crítico"   │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                           │                                  │
│  ┌───────────────────────▼─────────────────────────────┐    │
│  │  FASE 3 — VLM Asíncrono (cada 4s, hilo separado)    │    │
│  │  • Comprime frame anotado a 480x320 / JPEG 55%       │    │
│  │  • Envía a Qwen3-VL via Ollama API local             │    │
│  │  • Recibe JSON: {peligro, alerta_voz}                │    │
│  │  • Muestra alerta semántica en HUD                   │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                           │                                  │
│  ┌───────────────────────▼─────────────────────────────┐    │
│  │  FASE 4 — HUD + Benchmark                            │    │
│  │  • Dibuja texto: VLM alerta, FPS, latencia, riesgo  │    │
│  │  • Registra métricas en CSV IEEE                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              HILO AudioAlertManager (daemon)                  │
│  Cola FIFO → subprocess PowerShell SAPI5 → Parlante          │
└─────────────────────────────────────────────────────────────┘
```

### Diagrama de Hilos

```
Hilo Principal     ───── bucle YOLO + HUD (bloqueante)
Hilo Cámara        ───── captura continua sin buffer (daemon)
Hilo VLM           ───── Qwen3-VL c/4s (daemon, spawneado)
Hilo Audio         ───── cola TTS PowerShell (daemon)
```

---

## 3. Estructura de Archivos

```
scratch/
├── pipeline_vision.py        ← PIPELINE PRINCIPAL (recomendado)
│                               YOLOv8 rápido + Qwen VLM asíncrono
│
├── pipeline_vision_vlm.py    ← PIPELINE HÍBRIDO (síncrono)
│                               YOLO → Qwen en serie por frame
│                               Más semántico pero más lento
│
├── alert_manager.py          ← Motor de voz TTS
│                               PowerShell SAPI5 en hilo separado
│                               Cola con prioridad (descarta mensajes viejos)
│
├── benchmark_logger.py       ← Registrador de métricas IEEE
│                               Guarda FPS, latencia, RAM, objetos, peligro
│                               Exporta a CSV al finalizar
│
├── yolo_download.py          ← Script auxiliar de prueba Ollama
│
├── yolov8n.pt                ← Pesos YOLO nano (descargado automáticamente)
│                               El pipeline usa yolov8s.pt (small, más preciso)
│
└── venv/                     ← Entorno virtual Python
    └── Scripts/
        └── activate
```

---

## 4. Requisitos Previos

### Software (instalar antes)

| Software | Versión | Descarga | Para qué |
|---|---|---|---|
| Python | 3.12+ | [python.org](https://www.python.org/downloads/) | Lenguaje base |
| Ollama | Última | [ollama.com/download](https://ollama.com/download) | Servidor VLM local |
| IP Webcam | Cualquiera | Play Store (Android) | Cámara IP desde celular |

### Hardware mínimo

| Componente | Mínimo | Recomendado |
|---|---|---|
| CPU | 4 núcleos | 8 núcleos |
| RAM | 8 GB | 16 GB |
| GPU | No requerida | NVIDIA (CUDA) |
| Red WiFi | 2.4 GHz | 5 GHz (menos latencia) |

### Modelo Ollama requerido

```bash
# Descargar el VLM (2B parámetros, ~1.5 GB):
ollama pull qwen3-vl:2b

# Verificar que está disponible:
ollama list
```

---

## 5. Instalación Paso a Paso

### Paso 1 — Clonar / Descomprimir el proyecto

```powershell
# Si tienes el ZIP:
Expand-Archive -Path "scratch.zip" -DestinationPath "." -Force

# Entrar a la carpeta del proyecto
cd scratch
```

### Paso 2 — Crear el entorno virtual

```powershell
python -m venv venv
```

> **¿Por qué un entorno virtual?**  
> Aísla las dependencias del proyecto del Python global del sistema. Evita conflictos de versiones con otros proyectos.

### Paso 3 — Activar el entorno virtual

```powershell
# Windows PowerShell:
.\venv\Scripts\activate

# Verificar que está activo (debe aparecer "(venv)" al inicio del prompt):
# (venv) PS C:\...\scratch>
```

> ⚠️ **Error común:** Si ves "no se reconoce como cmdlet", estás en la carpeta incorrecta.  
> El `venv` y los `.py` están en `scratch\scratch\` no en `scratch\`.

### Paso 4 — Instalar dependencias

```powershell
pip install opencv-python ultralytics ollama pyttsx3 psutil pywin32
```

Esto instala automáticamente también: `torch`, `torchvision`, `numpy`, `pillow`, `matplotlib`, `scipy`, `httpx`, `pydantic` y sus dependencias.

**Tiempo estimado:** 3–8 minutos según velocidad de internet (descarga ~500 MB con PyTorch).

### Paso 5 — Verificar la instalación

```powershell
python -c "import cv2, ultralytics, ollama, pyttsx3, psutil; print('OK - Todo instalado')"
```

### Paso 6 — Verificar Ollama

```powershell
# En una terminal separada (debe quedar corriendo):
ollama serve

# En otra terminal, probar el modelo:
ollama run qwen3-vl:2b "Hola, ¿puedes ver imágenes?"
```

---

## 6. Configuración de la Cámara

### Opción A — Celular Android con IP Webcam (recomendado)

1. Instala **IP Webcam** (Pavel Khlebovich) desde Play Store — es gratuita
2. Abre la app → desplázate al fondo → toca **"Iniciar servidor"**
3. La app mostrará una URL como `http://192.168.1.X:8080`
4. Edita `pipeline_vision.py` línea 14:

```python
URL_VIDEO = "http://192.168.1.X:8080/video"  # ← Reemplaza X con tu IP
```

> **Importante:** PC y celular deben estar en la **misma red WiFi**.  
> Tu IP de red local puedes verla con: `ipconfig | Select-String "IPv4"`

### Opción B — Cámara USB / Webcam integrada

```python
URL_VIDEO = 0   # 0 = primera cámara detectada por el sistema
                # 1 = segunda cámara, etc.
```

### Verificar la conexión antes de ejecutar

```powershell
# Prueba que OpenCV puede conectarse:
python -c "
import cv2
cap = cv2.VideoCapture('http://192.168.1.X:8080/video')
ret, frame = cap.read()
print('Cámara OK:', ret, '| Resolución:', frame.shape if ret else 'N/A')
cap.release()
"
```

---

## 7. Cómo Ejecutar

### Ejecución normal (pipeline principal)

```powershell
# 1. Asegúrate que Ollama está corriendo en segundo plano
# (abre otra terminal y ejecuta: ollama serve)

# 2. Activa el entorno virtual
.\venv\Scripts\activate

# 3. Ejecuta el pipeline
python pipeline_vision.py
```

### Controles durante ejecución

| Tecla / Acción | Efecto |
|---|---|
| `ESC` | Detiene el sistema limpiamente |
| Cerrar ventana | También detiene el sistema |
| `Ctrl+C` en terminal | Fuerza el cierre (puede no guardar CSV) |

### Al finalizar

El sistema genera automáticamente el archivo `metricas_vision_ieee.csv` con todas las métricas registradas durante la sesión.

---

## 8. Módulos del Sistema

### `pipeline_vision.py` — Pipeline Principal

El archivo central del sistema. Implementa el flujo completo en un solo script con arquitectura de hilos.

**Constantes configurables:**

```python
MODEL_YOLO = "yolov8s.pt"           # Modelo YOLO (n/s/m/l/x)
URL_VIDEO = "http://IP:8080/video"  # URL de la cámara
VLM_MODEL = "qwen3-vl:2b"          # Modelo Ollama
UMBRAL_CONFIANZA = 0.45             # Confianza mínima YOLO (0.0–1.0)
```

**Función `capturar_camara_sin_latencia()`:**  
Corre en hilo separado. Lee frames del stream continuamente y siempre guarda el más reciente en `ultimo_frame`. Esto elimina el buffer acumulado de la cámara IP, garantizando que el pipeline siempre procese el frame más actual (latencia ~0ms de buffer).

**Función `analizar_escena_ollama(imagen_b64, contexto_yolo)`:**  
Corre en hilo daemon cada 4 segundos. Envía el frame anotado (con cajas YOLO) codificado en Base64 al modelo Qwen3-VL via API Ollama. Recibe un JSON estructurado con nivel de peligro y texto de alerta.

---

### `alert_manager.py` — Motor de Voz TTS

Gestiona las alertas de audio en un hilo daemon independiente para no bloquear el renderizado de video.

**Por qué PowerShell y no pyttsx3:**  
`pyttsx3` usa el motor SAPI5 de Windows via COM. Al ejecutarlo en un hilo secundario, el estado interno del engine COM se corrompe después de la primera llamada a `runAndWait()`, silenciando todas las alertas siguientes. La solución es usar PowerShell directamente, que lanza un proceso limpio e independiente por cada alerta.

```python
# Cada alerta ejecuta esto en un subproceso:
ps_script = """
    Add-Type -AssemblyName System.Speech;
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer;
    $s.Rate = 2;
    $s.Speak('Texto de la alerta');
    $s.Dispose();
"""
subprocess.run(["powershell", "-Command", ps_script], timeout=8)
```

**Sistema de prioridad de cola:**  
Si hay mensajes pendientes cuando llega uno nuevo, los anteriores se descartan. Esto garantiza que siempre se escuche la alerta más reciente (la más urgente) sin acumular mensajes obsoletos.

---

### `benchmark_logger.py` — Telemetría IEEE

Registra métricas de rendimiento por frame para análisis académico/paper.

**Métricas capturadas:**

| Campo | Descripción |
|---|---|
| `timestamp` | UNIX timestamp del frame |
| `fps` | Frames por segundo instantáneos |
| `inference_time_ms` | Latencia total de inferencia (ms) |
| `ram_mb` | RAM usada por el proceso (MB) |
| `objects_detected` | Número de objetos YOLO detectados |
| `event_type` | Descripción de la escena |
| `danger_level` | Alto / Medio / Bajo |

Al ejecutar `benchmark.export_csv()` (al cerrar el sistema), genera `metricas_vision_ieee.csv`.

---

### `pipeline_vision_vlm.py` — Pipeline Híbrido (alternativo)

Versión más lenta pero más semántica. A diferencia del pipeline principal, aquí **cada frame** pasa por YOLO y luego por Qwen3-VL de forma síncrona antes de mostrar el siguiente. Recomendado solo si se quiere evaluar la calidad de las respuestas VLM sin preocuparse por la latencia.

---

## 9. Lógica de Detección y Alertas

### Pseudo-profundimetría por Tamaño Relativo

Como la cámara no tiene sensor de profundidad (no es un LiDAR ni una cámara estéreo), el sistema estima la distancia midiendo cuánto espacio ocupa el objeto en el frame:

```
tamano_relativo = (ancho_caja × alto_caja) / (ancho_frame × alto_frame)
```

| Rango | Etiqueta | Riesgo resultante |
|---|---|---|
| `> 0.15` (15% del frame) | `CRITICO` | **Alto** |
| `0.04 – 0.15` (4–15%) | `Cerca` | **Medio** |
| `< 0.04` (< 4%) | `Lejos` | Bajo |

### Alertas de Audio Programadas

| Condición | Mensaje hablado | Frecuencia máxima |
|---|---|---|
| `riesgo == "Alto"` | *"Atención, riesgo alto detectado."* | Cada 3 segundos |
| `riesgo == "Medio"` (nuevo) | *"Precaución, objeto cercano."* | Cada 4 segundos |
| `riesgo == "Bajo"` (transición) | *"Camino despejado."* | Solo al cambiar de estado |
| `person CRITICO` detectada | *"Persona detectada, Riesgo Crítico."* | Cada 1.5 segundos |
| Respuesta VLM (Qwen3-VL) | Texto generado por la IA | Cada 4 segundos |

> El VLM **no repite alertas de "despejado"** para evitar fatiga auditiva (`if "despejad" not in alerta_texto.lower()`).

### HUD (Heads-Up Display)

El frame mostrado en pantalla incluye:

```
VLM Alerta: [texto del modelo Qwen3-VL]           ← Negro
Latencia YOLO: 134ms  |  7 FPS                     ← Azul oscuro
Riesgo Visual: Alto                                 ← Rojo/Naranja/Verde
```

---

## 10. Métricas y Telemetría IEEE

El archivo `metricas_vision_ieee.csv` generado tiene este formato:

```csv
timestamp,fps,inference_time_ms,ram_mb,objects_detected,event_type,danger_level
1748052123.4,7,134.5,312.4,2,"1 person CRITICO, 1 chair Cerca",Alto
1748052123.5,7,128.1,312.6,1,"1 person CRITICO",Alto
...
```

### Analizar con Python

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metricas_vision_ieee.csv")
print(df.describe())

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(df['inference_time_ms'])
plt.title('Latencia YOLO por frame (ms)')

plt.subplot(1,2,2)
df['danger_level'].value_counts().plot(kind='bar')
plt.title('Distribución de niveles de riesgo')
plt.tight_layout()
plt.show()
```

---

## 11. Solución de Problemas

### ❌ `ModuleNotFoundError: No module named 'ollama'`
El script se ejecutó con el Python del sistema, no el del entorno virtual.
```powershell
# Solución: activar el venv primero
.\venv\Scripts\activate
python pipeline_vision.py
```

### ❌ El audio solo suena la primera vez
Bug conocido de `pyttsx3` + COM en hilos de Windows. Ya corregido en esta versión usando PowerShell SAPI5 en `alert_manager.py`.

### ❌ La cámara no conecta (pantalla negra)
```powershell
# Verificar IP correcta:
ipconfig | Select-String "IPv4"

# Probar la URL en el navegador del PC:
# Abrir: http://192.168.1.X:8080/video

# Verificar que están en la misma red WiFi
```

### ❌ Ollama no responde (Error HTTP)
```powershell
# Verificar que Ollama está corriendo:
ollama list

# Si no está corriendo:
ollama serve

# Verificar que el modelo está descargado:
ollama pull qwen3-vl:2b
```

### ❌ FPS muy bajo (< 3 FPS)
- Reducir resolución de la cámara en la app IP Webcam (Settings → Video preferences)
- Cambiar `yolov8s.pt` por `yolov8n.pt` (más rápido, menos preciso)
- Aumentar el intervalo VLM de 4s a 8s:  
  `if time.time() - ultimo_tiempo_ollama > 8.0 and not analizando_ahora:`

### ❌ `.\venv\Scripts\activate` no reconocido
Estás en la carpeta equivocada. El venv está en `scratch\scratch\`:
```powershell
cd scratch   # entrar a la subcarpeta
.\venv\Scripts\activate
```

---

## Licencia

Proyecto académico — Universidad, Décimo Semestre, Laboratorio 3.  
Uso libre para fines educativos.
