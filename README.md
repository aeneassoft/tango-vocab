# Tango – Vokabeltrainer für Rioplatense-Spanisch

Audio zuerst, Text danach. Eine PWA fürs iPhone, komplett offline, FSRS-Wiederholung, drei Kartentypen (Wort, Lückensatz, Satz bauen). Wörter zuerst: Ein Lückensatz wird erst fällig, wenn sein Zielwort gefestigt ist (FSRS-Status „Review“) und alle anderen Listenwörter des Satzes schon als Einzelwörter gelernt wurden; „Satz bauen“ kommt, sobald mindestens drei Wörter gelernt sind. 500 Wörter in 7 Blöcken und über 300 Beispielsätze im Spanisch von Buenos Aires (`vos`, `acá`, `plata`, `colectivo`). Die Audio-Clips werden einmalig lokal mit F5-TTS aus einer Referenzstimme erzeugt und ins Repo committet.

**App:** https://aeneassoft.github.io/tango-vocab/

## Auf dem iPhone installieren

1. Die URL oben in Safari öffnen.
2. Teilen-Symbol → „Zum Home-Bildschirm" → „Hinzufügen".
3. Vom Home-Bildschirm öffnen (startet ohne Browserleiste) und einmal **Start** tippen.
4. Auf dem Startbildschirm kurz warten, bis „offline bereit" steht (die Audio-Clips werden im Hintergrund gecacht).
5. Ab jetzt läuft alles im Flugmodus – Wiederholungen, Audio, Statistik, Export.

## Wörter hinzufügen

1. Zeilen in `data/words.csv` (Verben als gesprochene Formen, IDs fortlaufend) und `data/sentences.csv` (jedes Wort in mindestens einem Satz) ergänzen.
2. `python tools/validate_data.py` – muss mit `OK` enden (prüft Dubletten, Rioplatense-Formen, Satzabdeckung und schreibt `app/data.json`).
3. `python tools/gen_audio.py` erzeugt nur die fehlenden Clips; dann `git add -A && git commit -m "Neue Wörter" && git push` – GitHub Pages veröffentlicht automatisch.

## Audio einmalig erzeugen (PC mit NVIDIA-GPU)

1. `bash tools/setup_tts.sh` – legt `.venv` an, installiert PyTorch mit CUDA, F5-TTS, Whisper und prüft ffmpeg (Windows: installiert es per winget).
2. Referenzstimme: `data/reference.wav` liegt im Repo (siehe unten). Eine eigene Stimme geht auch: 10–12 s ein einzelner Sprecher, wenig Hintergrund, möglichst viele `ll`/`y`-Wörter (calle, yo, ella, lluvia, playa) für den porteño-Klang; längere Dateien werden automatisch auf das beste Sprachstück gekürzt und mit Whisper transkribiert (`data/reference.txt`).
3. `python tools/gen_audio.py --test` – erzeugt Testclips in `data/test_clips/` (Einzelwörter in den Modi `bare` und `double`, drei Sätze).
4. `python tools/gen_audio.py` – alle fehlenden Clips nach `app/audio/`, AAC 48 kbps mono, −16 LUFS, plus `app/audio/manifest.json`. Tempo und Wortmodus sind als Standard hinterlegt (`--speed 0.85 --tempo 0.75 --word-mode double`).
5. `python tools/verify_audio.py` – hört alle Clips mit Whisper gegen und listet unverständliche; `--write-regen` merkt sie vor, `python tools/gen_audio.py --from-regen` erzeugt sie mit neuem Seed. `python tools/check_audio.py` spielt 10 zufällige Clips zum Selberhören.

### Referenzstimme und Lizenz

Die Stimme stammt aus dem Datensatz „Crowdsourced high-quality Argentinian Spanish speech data set“ (Google, [OpenSLR 61](https://www.openslr.org/61/), Lizenz [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)), Sprecher-ID 8784, vier Sätze zu 11,4 s zusammengeschnitten (`data/reference.wav`, Transkript `data/reference.txt`). Modell: F5-TTS mit dem spanischen Fine-Tune [jpgallegoar/F5-Spanish](https://huggingface.co/jpgallegoar/F5-Spanish) (Architektur F5TTS_Base, Checkpoint 1200000), NFE 32, CFG 2.0, Speed 0,85. Die Satz-Clips werden anschließend tonhöhenneutral auf 75 % Tempo gedehnt (ffmpeg `atempo`), weil F5-TTS das Sprechtempo der Referenz übernimmt und Anfänger langsamere Sätze brauchen. Einzelwörter werden im Modus `double` erzeugt (Wort zweimal synthetisiert, zweite Äußerung behalten), das vermeidet abgeschnittene Anlaute. Die erzeugten Clips in `app/audio/` sind abgeleitete Werke und stehen ebenfalls unter CC BY-SA 4.0.

Umfang: __AUDIO_STATS__

## Bekannte Einschränkungen

- Der Nur-hören-Modus pausiert, sobald das Display gesperrt wird (iOS lässt Web-Apps im Hintergrund kein Audio weiterspielen). Display anlassen.
- Die Spracheingabe (Web Speech API, `es-AR`) läuft über Apple-Server und braucht eine Internetverbindung; alles andere funktioniert offline.
- Das Gardel-Zitat („Volver") ist nicht enthalten: keine der ersten vier Zeilen besteht vollständig aus Wörtern der Liste. Eigene Songzeilen können als Satz mit `note=lyric` in `sentences.csv` ergänzt werden (die Längenregel gilt dann nicht).
- Ein neues Deployment ersetzt den App-Cache; Audio-Clips werden nur nachgeladen, wenn sich Dateiname oder Größe geändert hat.

## Struktur

```
data/words.csv        id,block,german,spanish,pos,note       Wortliste (500 Wörter, 7 Blöcke)
data/sentences.csv    id,word_ids,spanish,german,note        Beispielsätze
data/reference.wav    Referenzstimme (lokal, nicht im Repo)
tools/setup_tts.sh    venv + PyTorch/CUDA + F5-TTS + Whisper + ffmpeg
tools/gen_audio.py    F5-TTS-Pipeline → app/audio/*.m4a + manifest.json
tools/check_audio.py  Stichprobe anhören, Clips zum Neu-Erzeugen vormerken
tools/validate_data.py CSV-Prüfung → app/data.json
app/index.html        die gesamte App (eine Datei, Vanilla JS)
app/sw.js             Service Worker (cache-first, Audio-Abgleich, Update-Hinweis)
app/manifest.json     PWA-Manifest, app/icons/ Icons
.github/workflows/pages.yml   Push auf main → GitHub Pages aus app/
```
