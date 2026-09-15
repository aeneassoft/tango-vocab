# Auftrag: Spanisch-Vokabeltrainer „Tango" (PWA + lokale Audio-Generierung)

## Ziel

Baue in **einem Durchgang** einen Vokabeltrainer für mein iPhone. Der Trainer ist eine PWA (Progressive Web App), gehostet auf GitHub Pages, komplett offline nutzbar. Die Audio-Dateien werden **einmalig lokal** auf meinem PC (RTX 3080 Ti, 12 GB VRAM) mit F5-TTS aus einer Referenzstimme generiert und ins Repo committet. Danach fasse ich den PC nur noch an, wenn ich neue Wörter hinzufüge.

Ich lerne **Rioplatense-Spanisch** (Buenos Aires): `vos` statt `tú`, `acá` statt `aquí`, `plata` statt `dinero`, `colectivo` statt `autobús`, `lindo`, `dale`, `che`. Alle Beispielsätze in diesem Dialekt. Keine Formen aus Spanien oder Mexiko.

Arbeite selbstständig durch bis zum fertigen Ergebnis. Frag nur, wenn etwas wirklich blockiert (z. B. fehlende Referenzdatei). Am Ende gibst du mir die URL und eine kurze Zusammenfassung.

---

## Repo-Struktur

```
tango-vocab/
  data/
    words.csv            # Wortliste (siehe Format unten)
    sentences.csv        # Beispielsätze
    reference.wav        # meine Referenzstimme (lege ich ab; falls fehlt: STOPP und fragen)
  tools/
    setup_tts.sh         # venv + F5-TTS + torch (CUDA) installieren
    gen_audio.py         # F5-TTS-Loop: words + sentences → app/audio/*.m4a
    check_audio.py       # spielt 10 zufällige Clips ab, zeigt Text dazu
    validate_data.py     # prüft CSVs auf Dubletten, fehlende Felder, Nicht-Rioplatense-Formen
  app/
    index.html           # die gesamte App, EINE Datei, Vanilla JS, kein Framework, kein Build-Step
    sw.js                # Service Worker, cache-first, alles offline
    manifest.json        # standalone, Name „Tango", Icons
    icons/               # 192px + 512px, schlicht (schwarzer Hintergrund, weißes „T")
    audio/               # generierte Clips (werden committet)
  .github/workflows/
    pages.yml            # bei Push auf main: /app nach GitHub Pages
  README.md              # Installation auf dem iPhone in 5 Zeilen, Wörter hinzufügen in 3 Zeilen
```

---

## Datenformat

**data/words.csv** (UTF-8, Komma, Header):

```
id,block,german,spanish,pos,note
```

- `id`: fortlaufend, stabil (wird nie geändert, Audio-Dateiname = `w_<id>.m4a`)
- `block`: 1–7 (siehe unten)
- `german`: Prompt-Seite
- `spanish`: Zielform, **genau so wie gesprochen** (bei Verben die konjugierte Form, nicht der Infinitiv)
- `pos`: noun / verb / adj / adv / func / phrase
- `note`: optional, z. B. „vos-Form", „porteño für aquí"

**Verben werden als Formen gespeichert, nie als Infinitiv.** `tener` ergibt drei Zeilen: `tengo`, `tenés`, `tiene`. Bei den fünf Kernverben (ser, estar, tener, querer, ir) zusätzlich die Vergangenheitsform in der 1. Person (`fui`, `estuve`, `tuve`, `quise`, `fui`). Futur nur als `voy a` + Infinitiv. Kein Subjuntivo.

**data/sentences.csv**:

```
id,word_ids,spanish,german
```

- `word_ids`: durch `;` getrennte IDs der enthaltenen Wörter
- Jedes Wort soll in mindestens einem Satz vorkommen. Sätze kurz (4–9 Wörter), alltagsnah, Rioplatense.
- Audio-Dateiname = `s_<id>.m4a`

### Seed: die ersten 500 Wörter in 7 Blöcken

Fülle `words.csv` mit **500 Wörtern**, nach Blöcken geordnet, ungefähr in dieser Reihenfolge:

1. **Skelett / Funktionswörter** (~40): sí, no, y, o, pero, porque, con, sin, para, de, en, a, muy, más, menos, también, ahora, después, siempre, nunca, acá, allá, hoy, mañana, ayer, algo, nada, todo, otro, mismo, ya, todavía, casi, solo, bastante, demasiado …
2. **Kernverben als Formen** (~70): ser, estar, tener, querer, ir, poder, saber, hacer, decir, ver, dar, venir, salir, llegar, volver, esperar, mirar, escuchar, hablar, comer, tomar, comprar, pagar, buscar, entender, necesitar, gustar (me gusta / te gusta) — jeweils yo / vos / él-ella, plus `hay`.
3. **Fragewörter + Antworten** (~20): qué, quién, dónde, cuándo, cómo, cuánto, por qué, cuál, claro, dale, bueno, bárbaro, más o menos, ni idea …
4. **Stadt, Tag, Zahlen** (~90): casa, cuarto, calle, barrio, colectivo, subte, plata, comida, agua, café, pan, carne, vino, cerveza, mañana/tarde/noche, hora, semana, mes, año, Wochentage, Monate, Zahlen 0–100 (als Einzelwörter bis 20, dann Zehner), cerca, lejos, izquierda, derecha, adelante, atrás …
5. **Menschen + Körper** (~60): hombre, mujer, chico, chica, señor, señora, amigo, amiga, gente, familia, padre, madre, hermano, hermana, nombre, cabeza, ojo, mano, brazo, pie, pierna, pecho, espalda, hombro, peso …
6. **Milonga** (~60): bailar, caminar, abrazo, pista, mesa, tanda, cortina, orquesta, letra, compás, pausa, milonga, vals, tango, cabeceo, mirada, permiso, gracias, ¿bailás?, un tango más, despacio, escuchá, todavía no, disculpá, lindo, sentir, seguir, llevar, girar, cruzar, parar …
7. **Urteile, Gefühle, Alltag** (~160): bien, mal, bueno, malo, lindo, feo, grande, chico, cansado, tranquilo, contento, triste, difícil, fácil, verdad, mentira, nuevo, viejo, caro, barato, caliente, frío, rápido, lento, temprano, tarde, abierto, cerrado, lleno, vacío, sowie Alltagsnomen (puerta, mesa, silla, cama, luz, ropa, zapato, camisa, saco, teléfono, llave, papel, libro, música, trabajo, clase, profesor, médico, hospital, farmacia, banco, mercado, tienda, bar, restaurante …) und Höflichkeit (por favor, perdón, de nada, mucho gusto, hasta luego, chau, buen día, buenas noches …)

Erzeuge dazu **mindestens 250 Sätze**, so dass jedes Wort mindestens einmal in einem Satz vorkommt. Außerdem die ersten vier Zeilen von Gardels „Volver" als Sätze mit Markierung `note=lyric` (nur die Zeilen, die vollständig aus Wörtern der Liste bestehen).

`validate_data.py` muss anschlagen bei: doppelten `spanish`-Werten im selben Block, leeren Feldern, Infinitiven in `pos=verb`, sowie bei den Wörtern `aquí`, `dinero`, `autobús`, `vosotros`, `coche`, `ordenador`, `móvil`, `tú`, `tienes`, `quieres`, `puedes` (Fehler mit Hinweis auf die Rioplatense-Form).

---

## Audio-Pipeline (lokal, einmalig)

**tools/setup_tts.sh**: Python-venv anlegen, torch mit CUDA passend zur installierten Treiberversion, F5-TTS (`pip install f5-tts`), ffmpeg prüfen. Danach `python -c "import torch; print(torch.cuda.is_available())"` muss `True` ausgeben. Falls torch/CUDA-Mismatch: selbstständig beheben.

**tools/gen_audio.py**:
- Lädt `data/reference.wav` (meine Referenzstimme, 60–120 s, ein alter Porteño). Falls die Datei fehlt: abbrechen mit klarer Meldung. Falls länger als 30 s: automatisch die ersten 20–25 s mit Sprache (nicht Stille) als Referenz-Snippet extrahieren, weil F5-TTS mit kurzen Referenzen besser arbeitet.
- Referenztext: Wenn `data/reference.txt` existiert, nutzen; sonst mit Whisper (large-v3, lokal, GPU) transkribieren und als `reference.txt` speichern.
- Loop über `words.csv` und `sentences.csv`. Für jedes Item: F5-TTS-Inferenz → WAV → mit ffmpeg zu AAC 48 kbps mono `.m4a` → `app/audio/w_<id>.m4a` bzw. `s_<id>.m4a`.
- Stille am Anfang/Ende trimmen, Lautstärke auf −16 LUFS normalisieren.
- Bereits vorhandene Dateien überspringen (`--force` zum Neugenerieren, `--ids 12,13` für einzelne).
- Einzelwörter mit leichter Satzeinbettung generieren, wenn F5 bei isolierten Wörtern abschneidet (z. B. „… calle." mit anschließendem Trimmen) — teste beides, nimm was sauberer klingt.
- Am Ende: `app/audio/manifest.json` mit allen Dateinamen + Dauer für den Service Worker.

**tools/check_audio.py**: spielt 10 zufällige Clips ab (per `ffplay` oder `simpleaudio`), zeigt Text dazu, fragt „ok / neu". „neu" schreibt die ID in `data/regen.txt`, die `gen_audio.py --from-regen` abarbeitet.

Wichtig für die Aussprache: Das Ziel ist der **porteño-Klang** — `ll`/`y` als „sch", `vos`-Intonation. Wenn Clips zu neutral-lateinamerikanisch klingen, teste eine längere Referenz mit vielen `ll`/`y`-Wörtern. Notiere im README, welche Referenz-Einstellung am Ende verwendet wurde.

---

## App-Verhalten (app/index.html)

**Grundprinzip: Audio zuerst, Text danach. Produktion, nicht Wiedererkennung.**

### Scheduling
- FSRS-Algorithmus (portiere `ts-fsrs` in eine einzelne JS-Funktion oder binde es als inline-Script ein; kein Build-Step, kein npm zur Laufzeit).
- Zustand pro Karte in **IndexedDB** (Wrapper mit `idb`-artiger Hilfsfunktion, inline). Kein Server, kein Login.
- Tageslimit: **7 neue Wörter** + alle fälligen Wiederholungen. Neue Wörter in Block-Reihenfolge.
- Vier Bewertungen: Again / Hard / Good / Easy. Bei getippter/gesprochener Antwort: automatisch Good bei Treffer, Again bei Fehlschlag; Hard/Easy per Tastendruck überschreibbar.

### Kartentypen (jedes Wort hat alle drei, sie werden unabhängig gescheduled)
1. **Wort**: Audio spielt automatisch → nach 1,5 s erscheint das deutsche Wort → ich tippe oder spreche die spanische Form → Abgleich (siehe unten).
2. **Lückensatz**: Audio des ganzen Satzes → Satz erscheint mit Lücke (das Zielwort) + deutsche Übersetzung → ich fülle die Lücke.
3. **Satz bauen**: drei zufällige Wörter aus dem gelernten Bestand → ich spreche/tippe einen Satz → **Selbstbewertung** (kein Abgleich), Freitext wird im Log gespeichert.

### Antwortabgleich
- Normalisierung: Kleinschreibung, Akzente **werden geprüft** (á ≠ a), aber Satzzeichen/Whitespace ignoriert. Bei `¿bailás?` reicht `bailás`.
- Tippfehler-Toleranz: Levenshtein ≤ 1 bei Wörtern ab 6 Buchstaben → „fast" (zählt als Hard, zeigt die korrekte Form).
- Spracheingabe: Web Speech API (`webkitSpeechRecognition`), `lang = "es-AR"`, Fallback `es-419`. Ergebnis wird wie getippt behandelt. Wenn Spracherkennung nicht verfügbar: Button ausblenden, nicht crashen.

### Bedienung
- Mobile-first, Portrait, iPhone-Safari. Große Touchflächen. Dunkles Design (schwarz, warmes Weiß, ein Akzent in Dunkelrot). Keine Animationen außer einem kurzen Fade.
- Tippen auf die Karte = Audio erneut abspielen.
- Startbildschirm: „Heute: X neu, Y fällig" + Start-Button + Streak-Zähler (Tage mit ≥1 Review) + Gesamtwörter gelernt.
- Ende der Session: kurze Statistik (richtig/falsch, mittlere Latenz).
- Einstellungen: Tageslimit neue Wörter (Default 7), Auto-Play an/aus, Blöcke aktivieren/deaktivieren, **Export** (gesamtes Review-Log + Kartenzustände als JSON in die Dateien-App), **Import** (JSON zurückspielen), **Reset** mit Sicherheitsabfrage.
- „Nur hören"-Modus: spielt fällige Wörter/Sätze nacheinander ab ohne Abfrage (für Gehen/Bus), 0,8 s Pause zwischen Clips.

### Logging
- Jede Review: `{ts, card_id, type, latency_ms, correct, rating, input}` in IndexedDB.
- Wöchentliche Zusammenfassung im Statistik-Screen: Wörter pro Block gesichert, mittlere Latenz-Trend, Fehlerliste.

### Offline / PWA
- `manifest.json`: `display: standalone`, `name: "Tango"`, `short_name: "Tango"`, `theme_color: #000`, `background_color: #000`, Icons.
- `sw.js`: bei Install alles aus `app/audio/manifest.json` + `index.html` + `manifest.json` cachen. Strategie **cache-first**. Versionsstring im SW; bei neuem Deploy alte Caches löschen und neue Audio-Dateien nachladen. In der App ein dezenter Hinweis „Update verfügbar – neu laden".
- Muss im Flugmodus vollständig funktionieren, nachdem die App einmal geöffnet wurde.
- iOS-Eigenheiten beachten: Audio-Autoplay erst nach erster Nutzerinteraktion (Start-Button reicht), `apple-mobile-web-app-capable`, `viewport-fit=cover`, Safe-Area-Insets.

---

## Deploy

- `.github/workflows/pages.yml`: bei Push auf `main` den Ordner `app/` nach GitHub Pages veröffentlichen (Actions-basiertes Pages-Deployment, kein `gh-pages`-Branch).
- Repo per `gh` CLI anlegen (privat ist ok, Pages funktioniert bei privaten Repos nur mit Pro — falls nicht vorhanden: public, das Repo enthält nichts Persönliches).
- README: (1) Installation auf dem iPhone: URL in Safari öffnen → Teilen → „Zum Home-Bildschirm" → einmal öffnen, Start drücken, dann ist alles offline. (2) Wörter hinzufügen: Zeilen in `words.csv`/`sentences.csv` ergänzen → `python tools/validate_data.py` → `python tools/gen_audio.py` → commit + push.

---

## Reihenfolge der Arbeit

1. Repo-Skelett, CSVs mit den 500 Wörtern + 250 Sätzen, `validate_data.py` läuft grün.
2. `setup_tts.sh` ausführen, CUDA verifizieren, F5-TTS mit einem Testsatz gegen `reference.wav` prüfen. **Hier kurz stoppen und mir 3 Test-Clips zum Anhören zeigen** (Pfad nennen), bevor du 750 Dateien generierst.
3. Nach meinem OK: `gen_audio.py` komplett, `check_audio.py` einmal durchlaufen.
4. `index.html`, `sw.js`, `manifest.json`, Icons. Lokal mit `python -m http.server` testen; prüfe im Browser: Karte erscheint, Audio spielt, Antwort wird abgeglichen, IndexedDB speichert, Reload behält Zustand.
5. Workflow, Push, Pages aktivieren, URL prüfen.
6. Abschlussbericht: URL, Anzahl Wörter/Sätze/Clips, Gesamtgröße von `app/audio`, verwendete Referenz-Einstellung, bekannte Einschränkungen.

---

## Abnahmekriterien

- App startet auf dem iPhone vom Home-Bildschirm ohne Browserleiste.
- Flugmodus: Session mit Audio läuft vollständig durch.
- Erste Karte spielt Audio **vor** dem Text.
- Spracheingabe erkennt `calle` und `plata` (es-AR) korrekt.
- `tenés`, `sos`, `acá`, `dale` sind im Bestand; `tú`, `aquí`, `dinero` sind es nicht.
- Export erzeugt eine JSON-Datei, Import stellt den Zustand her.
- Gesamtgröße `app/audio` < 15 MB.
- Neue Wörter hinzufügen dauert unter 5 Minuten (CSV → validate → gen → push).

Keine Frameworks, kein Bundler, keine externen CDNs zur Laufzeit (alles inline, damit es offline ist). Kommentare im Code auf Englisch, UI-Texte auf Deutsch, Lerninhalt auf Rioplatense-Spanisch.
