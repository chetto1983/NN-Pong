# Pong Physics and Presentation Implementation Plan

> **For agentic workers:** Use test-driven execution, independent file ownership, and verification before completion.

**Goal:** Correggere urti e temporizzazione della pallina e consegnare una grafica Pong pulita.

**Architecture:** Package `pong_game` con configurazione, rete, simulazione e app; `pong.py` mantiene CLI e compatibilità degli import.

**Tech Stack:** Python 3.12+, NumPy, Pygame e unittest.

**Spec:** `docs/superpowers/specs/2026-11-03-pong-physics-design.md`.

## Contratto condiviso

- `PongSimulation(seed=0, opponent_skill=0.82, trace=False)`; `step(dt=1/60, control=True)`.
- Coordinate/velocità: `ball_x/y`, `ball_vx/vy` in px/s, `player_y`, `opponent_y`, `player_vy`, `opponent_vy`, `ball_spin`.
- `stats` conserva frames/hits/misses/player_score/opponent_score/direction_changes; `elapsed` in secondi.
- `prediction`, `raw_prediction`, `fold`, `output`; `predict_player_intercept()`, `reset()`, `serve(direction)`, `scale_ball_speed(factor)`.
- `PongApp(seed=42)`: `screen`, `game`, `paused`, `running`, `show_graph=False`, `handle_event`, `update(dt)`, `draw()`, `run()`.
- Il renderer importa config/rete/simulazione direttamente dal package, senza importare `pong.py`.

## Review Focus

- Contatti fisici al tempo corretto, senza correzioni che riportino una palla mancata davanti alla racchetta.
- Raggio palla ed estremità racchetta coerenti con il disegno.
- Tempo fisico indipendente dal rendering, inclusi pausa e riavvio.
- Nessun incremento arbitrario della velocità; input +/- e movimento racchetta spiegano le variazioni.
- Diagnostica fuori dal campo e controlli utilizzabili dopo un ridimensionamento.

## Task 1: Fisica (worker)

- [x] Creare config/rete/simulazione e test di regressione prima delle correzioni.
- [x] Testare urti continui, pareti, estremità, movimento, punteggi e previsione.
- [x] Aggiornare le fixture fisiche dei test esistenti alle unità px/s.

## Task 2: Presentazione (worker)

- [x] Ridisegnare l'app su `pong_game/app.py` e aggiornare `tests/test_app.py`.
- [x] Verificare rendering, controlli, tempo fisso e diagnostica separata.

## Task 3: Integrazione (coordinatore)

- [x] Aggiornare `pong.py`, README e documentazione fisica.
- [x] Suite completa, stress multi-seed e verifica Windows con screenshot.
- [x] Revisione indipendente, eventuali correzioni, commit locale.

## Evidenze di esecuzione

- Suite corrente: 92 test passati, inclusi 24 regressioni fisiche Pong,
  15 test dell'app Pong e la suite Snake.
- `python pong.py --test`: 50.000 campioni, massimo errore `2.274e-13`.
- Stress finale con battuta avversaria: 100 seed × 10.000 passi = 1.000.000;
  3.019 respinte, zero mancate, punteggio aggregato 38–0.
- Fuzz del worker: 3.000 traiettorie fino a 20.000 px/s concordano tra passo
  unico e suddivisione in sette tempi entro `2.75e-7`.
- Prova Windows nativa: 240 frame renderizzati, tempo fisico 4 s, pausa,
  ripresa, diagnostica, ridimensionamento, click, reset e uscita.
- Test schermo 1366×768: finestra ridotta all'avvio, footer accessibile,
  ritorno da schermo intero alla misura iniziale.
- Catture finali: `docs/images/pong.png`, `docs/images/pong-diagnostics.png`;
  originale acquisito in `output/playwright/pong-before.png`.

## Decisioni durante l'esecuzione

- Racchette perfettamente passive dissipavano la componente verticale in
  scambi piatti. Aggiunta una battuta avversaria con moto sinusoidale reale;
  l'effetto sulla palla resta calcolato dagli impulsi del modello fisico.
- Il target raggiungibile dell'avversario viene limitato prima di aggiungere
  la battuta, così può allontanarsi da una parete anche con intercetti estremi.
- La finestra iniziale si adatta al desktop invece di imporre 1200×900 fisici;
  il canvas logico e le proporzioni restano invariati.
- Revisione indipendente di codice e immagini senza rilievi aperti; ulteriori
  240 contatti simultanei verificati dal revisore.
- Un controllo ulteriore del coordinatore ha riprodotto un blocco per palla
  compressa esattamente fra calotta e sponda. Corretto con arresto del motore
  e risposta vincolata, senza scartare tempo o spostare la palla. Test a timeout
  su 12 casi specchiati e quasi assiali, più quattro urti laterali che devono
  mantenere il movimento e contare una sola respinta. Regressioni RED→GREEN.
