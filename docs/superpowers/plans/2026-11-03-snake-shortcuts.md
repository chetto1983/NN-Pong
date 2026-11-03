# Snake Shortcuts Implementation Plan

> **For agentic workers:** Follow test-first execution, separate file ownership, and verify before completion.

**Goal:** Ridurre le mosse di Snake tramite una rete analitica completa che osserva mela e corpo.

**Architecture:** Nuova rete sparse ReLU e JSON v2, protocollo comune di osservazione,
integrazione nelle regole/CLI/app esistenti e confronto riproducibile con il ciclo.

**Tech Stack:** Python 3.12+, NumPy, Pygame, unittest; nessuna nuova dipendenza.

**Spec:** `docs/superpowers/specs/2026-11-03-snake-shortcuts-design.md`.

## Review Focus

- Nessun calcolo delle mosse sicure o fallback in Python durante l'inferenza.
- Conservazione dell'ordine del corpo, nessun sorpasso della mela o della coda.
- Coda liberata correttamente, wrap, crescita, ultime celle e stati senza mosse.
- Validazione dei file sparse, versioni precedenti, conteggio onesto dei parametri.
- CLI e grafica non devono continuare a presentare la politica come cieca alla mela.

## Task 1: Rete (worker)

- [x] Test RED per osservazione, logits/gate, corpo/coda e JSON.
- [x] Implementare `snake_game/shortcut_network.py`, `weights/snake_shortcuts.json`
  e `tests/test_snake_shortcuts.py`.
- [x] Confrontare i logits con un oracolo indipendente e provare partite complete.

## Task 2: Integrazione (coordinatore)

- [x] Protocollo comune, selezione e caricamento delle reti, compatibilità baseline.
- [x] Simulazione predefinita con scorciatoie, CLI `--policy` / `--compare`, UI aggiornata.
- [x] Test integrazione, benchmark 0–99 e holdout 100–199, documentazione e artefatti.
- [x] Verifica nativa e revisione indipendente; correzioni e commit locale.

## Risultato

- Architettura 1008→955→28→11→20→4, 7.166 coefficienti memorizzati,
  991.006 parametri densi equivalenti; nessun training o RL.
- 115 test passati. Revisione indipendente senza rilievi sul codice.
- Seed 0–99: media 15.465,24→8.369,72, riduzione 45,88%.
- Holdout 100–199: media 15.427,51→8.380,99, riduzione 45,68%.
- 200 completamenti su 200 per ciascuna politica, zero collisioni/timeout.
- Prova grafica Windows: 600 frame/100 mosse, quattro mele, 22 scorciatoie,
  ciclo di riferimento, aiuto, pausa, riavvio e uscita.
- I tempi CPU sono maggiori della baseline: beneficio nelle mosse, non nel
  costo dell'inferenza. Protocollo completo in `docs/benchmarks/README.md`.
