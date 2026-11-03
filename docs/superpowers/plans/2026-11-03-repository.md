# NN Pong Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Consegnare una repository locale eseguibile e verificata, eliminando il generatore originale.

**Architecture:** Estrarre il gioco in `pong.py`; mantenere simulazione, rete e grafica esistenti. Separare test e documentazione, correggendo il fold rispetto al raggio della palla.

**Tech Stack:** Python 3.12+, NumPy 2.5.4, Pygame 2.6.1, unittest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-repository-design.md`

## Global Constraints

- Python 3.12 o superiore; NumPy 2.5.4 e Pygame 2.6.1.
- Nessun training; cinematica esterna alla rete conservata.
- File originali recuperabili tramite Git, ambiente virtuale escluso dai commit.

## Review Focus

- Rimbalzi sulle pareti: rete e simulazione devono usare lo stesso centro palla.
- Conteggi CLI nulli o negativi: errore esplicito, nessuno stress test vuoto.
- Avvio da Windows: nessun percorso `/mnt/data` e installazione locale documentata.
- Grafica senza display in CI: rendering e uscita tramite driver SDL dummy.
- Regressioni di simulazione: punteggi, riproducibilità e modalità headless.

## Task 1: Gioco e file della repository

**Files:** `pong.py`, `tests/test_pong.py`, `tests/test_app.py`, `requirements.txt`, `README.md`, `.gitignore`, `.gitattributes`, `.github/workflows/tests.yml`.

**Interfaces:** mantenere `PongNetV5.forward(paddle_y, y_pred)`, `PongSimulation.step()`, `PongApp.run()`, `test_network()` e `stress_test(seeds, frames)`.

- [ ] Estrarre senza modifiche il codice dalla stringa originale; avvio precedente già verificato fallente con FileNotFoundError.
- [ ] Scrivere e provare test di regressione: `forward(325, 858.4166666667)` deve produrre fold `409.5833333333`; CLI con zero/negativi deve uscire con codice 2.
- [ ] Correggere limiti del fold e validazione CLI; eseguire `python -m unittest discover -s tests -v`, atteso zero errori.
- [ ] Aggiungere istruzioni, dipendenze bloccate e workflow CI.
- [ ] Provare `python pong.py --test`, attesi tre PASS; stress test 100 seed x 10000 frame e rendering Windows.
- [ ] Eliminare `pongNN.py` dopo la verifica, richiedere revisione indipendente e registrare il risultato in Git.

## Esecuzione

- Baseline: `3d5d756`, originale conservato nella cronologia.
- Il progetto nasce nella cartella esplicitamente richiesta; esecuzione in questa cartella senza worktree aggiuntivo.
