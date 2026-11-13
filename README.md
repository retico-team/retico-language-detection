# retico-language-detection

A Retico module for automatically detecting the input language from the previous module as text or audio, and forward it to the next module.

[Link to Video Tutorial in Google Drive](https://drive.google.com/file/d/1EOFs2QZAImt_sra9D3LY93pz7pg8ggDG/view?usp=sharing)

## Installation and Requirements

To use the language detection module you first need to install the retico-language-detection package:
* Install retico-language-detection:
```pip install retico-language-detection```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-language-detection.git```

The audio language model ([speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)) is downloaded from Hugging Face the first time the module is imported.

## Modules

### `LanguageDetectionModule`
Subscribes to `TextIU` or `AudioIU` and outputs a `LanguageDetectionIU`: a copy of the input with a `language` attribute (an ISO 639-1 code, e.g. `"en"`) and a `confidence` attribute. Modules further down the network can use `language` to switch languages automatically, e.g. `GoogleASRModule` from [retico-googleasr](https://github.com/retico-team/retico-googleasr) or `MultilingualTTSModule` from [retico-multilingual-tts](https://github.com/retico-team/retico-multilingual-tts).

* **Text** is classified with [Lingua](https://github.com/pemistahl/lingua-py), using the last 10 text IUs as context.
* **Audio** is classified with SpeechBrain's VoxLingua107 model, about once per second. The language reported is the most common one over the recent predictions, to smooth out occasional misclassifications.

It takes no arguments.

### Example
**Note:** The example uses `MicrophoneModule`, `SpeakerModule` and `DebugModule` from `retico_core`, `GoogleASRModule` from [retico-googleasr](https://github.com/retico-team/retico-googleasr) (`pip install retico-googleasr`) and `MultilingualTTSModule` from [retico-multilingual-tts](https://github.com/retico-team/retico-multilingual-tts) (`pip install retico-multilingual-tts`). The language of your speech is detected from the audio, so the ASR transcribes in that language; it is detected again from the transcribed text, so the TTS answers in that language.

```python
from retico_core import network
from retico_core.audio import MicrophoneModule, SpeakerModule
from retico_core.debug import DebugModule
from retico_googleasr import GoogleASRModule
from retico_multilingual_tts import MultilingualTTSModule
from retico_language_detection import LanguageDetectionModule


mic = MicrophoneModule(rate=16000, frame_length=0.2)
lgr_audio = LanguageDetectionModule()
asr = GoogleASRModule(rate=16000)
lgr_text = LanguageDetectionModule()
tts = MultilingualTTSModule()
spk = SpeakerModule(rate=22050)
debug = DebugModule()

mic.subscribe(lgr_audio)
lgr_audio.subscribe(asr)
asr.subscribe(lgr_text)
lgr_text.subscribe(tts)
tts.subscribe(spk)

lgr_text.subscribe(debug)

network.run(mic)

input("Running...\n")

network.stop(mic)
```

### Example 2
**Note:** The example detects the language of typed text, using `KeyboardModule` from [retico-keyboard](https://github.com/retico-team/retico-keyboard) (`pip install retico-keyboard`).

```python
from retico_core import network
from retico_core.debug import DebugModule
from retico_keyboard import KeyboardModule
from retico_language_detection import LanguageDetectionModule


keyboard = KeyboardModule()
lgr = LanguageDetectionModule()
debug = DebugModule()

keyboard.subscribe(lgr)
lgr.subscribe(debug)

network.run(keyboard)

input("Running...\n")

network.stop(keyboard)
```
