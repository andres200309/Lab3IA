import threading
import queue
import subprocess

class AudioAlertManager:
    """Gestor de audio usando PowerShell SAPI5 nativo.
    
    Solución robusta: cada alerta lanza un proceso PowerShell independiente,
    evitando el bug de pyttsx3/COM que se bloquea tras la primera llamada.
    """
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[Audio] Motor TTS PowerShell iniciado.")

    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                # PowerShell SAPI5 nativo — siempre funciona en Windows
                ps_script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = 2; "          # -10 (lento) a 10 (rápido), 2 = normal-rápido
                    f"$s.Speak('{text}');"
                    f"$s.Dispose();"
                )
                subprocess.run(
                    ["powershell", "-Command", ps_script],
                    timeout=8,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.TimeoutExpired:
                print(f"[Audio] Timeout en alerta: {text}")
            except Exception as e:
                print(f"[Audio] Error TTS: {e}")
            finally:
                self.q.task_done()

    def enviar_alerta(self, texto):
        """Encola un mensaje para ser convertido a audio.
        Si hay mensajes pendientes, los descarta para priorizar el más reciente.
        """
        if not self.q.empty():
            with self.q.mutex:
                self.q.queue.clear()
        self.q.put(texto)
        print(f"[Audio] >> VOZ: {texto}")
