from retico_core import AbstractTriggerModule, UpdateMessage, UpdateType, network
from retico_core.audio import MicrophoneModule, SpeakerModule
from retico_core.debug import CallbackModule
from retico_core.text import TextIU
from retico_googleasr import GoogleASRModule
from retico_multilingual_tts import MultilingualTTSModule

from language_detection import LanguageDetectionModule

import os
import threading
import time

class SimpleTerminalInputModule(AbstractTriggerModule):
    @staticmethod
    def name():
        return "Terminal Input Module"

    @staticmethod
    def description():
        return "Reads text input from the terminal and outputs it as TextIU."

    @staticmethod
    def input_ius():
        return []

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                os.system("cls" if os.name == "nt" else "clear")
                line = input("Enter text: ")
            except (EOFError, KeyboardInterrupt):
                self._stop_event.set()
                break
            text = line.strip()
            if not text:
                continue
            self.trigger({"text": text}, UpdateType.COMMIT)

    def trigger(self, data={}, update_type=UpdateType.ADD):
        text = data.get("text", "")
        iu = TextIU(self, iuid=0, payload=text)
        um = UpdateMessage()
        um.add_iu(iu, update_type)
        self.append(um)

def callback(update_msg):
    for iu, ut in update_msg:
        if hasattr(iu, 'language') and iu.language:
            print(f"Detected language: '{iu.language}' with confidence {iu.confidence:.2f}")
        else:
            text = getattr(iu, 'text', iu.payload if hasattr(iu, 'payload') else None)
            print(f"{ut}: {text}")

if __name__ == "__main__":

    # ---- Text ---- #

    # ter = SimpleTerminalInputModule()
    # lgr = LanguageDetectionModule()
    # debug = CallbackModule(callback)
    
    # ter.subscribe(lgr)
    # lgr.subscribe(debug)
    
    # network.run(ter)
    
    # print("Running...\n")

    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     network.stop(ter)
    
    # ---- Audio ---- #
    
    mic = MicrophoneModule(rate=16000, frame_length=0.2)
    lgr_audio = LanguageDetectionModule()
    asr = GoogleASRModule(rate=16000)
    lgr_text = LanguageDetectionModule()
    tts = MultilingualTTSModule()
    spk = SpeakerModule(rate=22050)
    debug = CallbackModule(callback)
    
    mic.subscribe(lgr_audio)
    lgr_audio.subscribe(asr)
    asr.subscribe(lgr_text)
    lgr_text.subscribe(tts)
    tts.subscribe(spk)
    
    asr.subscribe(debug)
    
    network.run(mic)
    
    input("Running...\n")
    
    network.stop(mic)