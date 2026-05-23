import time
import psutil
import csv
import os

class BenchmarkLogger:
    """Clase para registrar métricas de hardware/software orientado a tu paper IEEE."""
    def __init__(self, filename="ieee_metrics_log.csv"):
        self.filename = filename
        self.metrics = []
        self.process = psutil.Process(os.getpid())

    def record_frame(self, fps, inference_time_ms, yolo_objs_count, danger_level, event_type):
        """Registra la telemetría del frame actual."""
        mem_mb = self.process.memory_info().rss / (1024 ** 2)
        self.metrics.append({
            "timestamp": time.time(),
            "fps": round(fps, 2),
            "inference_time_ms": round(inference_time_ms, 2),
            "ram_mb": round(mem_mb, 2),
            "objects_detected": yolo_objs_count,
            "event_type": event_type,
            "danger_level": danger_level
        })

    def export_csv(self):
        """Guarda todas las métricas en un archivo CSV apto para graficar en reportes."""
        if not self.metrics:
            print("No hay métricas para exportar.")
            return
            
        keys = self.metrics[0].keys()
        with open(self.filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.metrics)
        print(f"\n[OK] Métricas IEEE exportadas correctamente a {self.filename}")
