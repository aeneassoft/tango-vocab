# Tango – Vokabeltrainer für Rioplatense-Spanisch

Audio zuerst, Text danach. Eine PWA fürs iPhone, komplett offline, FSRS-Wiederholung, drei Kartentypen (Wort, Lückensatz, Satz bauen). 500 Wörter in 7 Blöcken und über 300 Beispielsätze im Spanisch von Buenos Aires (`vos`, `acá`, `plata`, `colectivo`). Die Audio-Clips werden einmalig lokal mit F5-TTS aus einer Referenzstimme erzeugt und ins Repo committet.

**App:** __APP_URL__

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
2. Referenzstimme als `data/reference.wav` ablegen: 60–120 s ein einzelner Sprecher, wenig Hintergrund, möglichst viele `ll`/`y`-Wörter (calle, yo, ella, lluvia, playa) für den porteño-Klang. Die Datei bleibt lokal (`.gitignore`).
3. `python tools/gen_audio.py --test` – erzeugt Testclips in `data/test_clips/` (Einzelwörter in den Modi `bare` und `double`, drei Sätze) und transkribiert die Referenz mit Whisper nach `data/reference.txt`.
4. `python tools/gen_audio.py` (bzw. `--word-mode double`, wenn die double-Variante sauberer klingt) – alle Clips nach `app/audio/`, AAC 48 kbps mono, −16 LUFS, plus `app/audio/manifest.json`.
5. `python tools/check_audio.py` – spielt 10 zufällige Clips; „neu" merkt sie in `data/regen.txt` vor, `python tools/gen_audio.py --from-regen` erzeugt sie neu.

Verwendete Referenz-Einstellung: __REFERENCE_SETTING__

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
