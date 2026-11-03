# NN Pong

Pong automatico in Python e Pygame: la racchetta verde è controllata da una
piccola rete ReLU con pesi costruiti analiticamente, senza addestramento.
La racchetta gialla usa una strategia deterministica di confronto.

L'idea nasce da [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
di Rikiya Takehi, Jayden Teoh, Alex Zhang e Omar Khattab. Questo progetto è una
dimostrazione indipendente ispirata al principio di esprimere un algoritmo nei
pesi di una rete; non riproduce i benchmark dell'articolo.

## Requisiti e avvio

Python **3.12 o superiore**. Versioni delle dipendenze bloccate in
`requirements.txt`; ambiente locale verificato con Python 3.13 su Windows.

Da PowerShell, nella cartella del progetto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pong.py
```

Non serve attivare l'ambiente virtuale. Se `.venv` è già stato preparato, basta
l'ultimo comando. Su Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pong.py
```

Entrambe le racchette sono automatiche. La finestra mostra punteggio, uscita
della rete, previsione, respinte e palle mancate.

| Tasto | Azione |
| --- | --- |
| `Esc` | Esci |
| `Spazio` | Pausa / riprendi |
| `R` | Ricomincia la partita |
| `G` | Mostra / nascondi il grafico del fold |
| `+` oppure `=` | Aumenta la velocità della palla |
| `-` | Riduci la velocità della palla |

## Verifiche

Con l'ambiente attivato, oppure sostituendo `python` con il percorso indicato sopra:

```bash
# Test di regressione, CLI e rendering Pygame senza finestra
python -m unittest discover -s tests -v

# Controllo numerico della rete su 50.000 campioni
python pong.py --test

# Simulazione riproducibile: un milione di frame, senza finestra
python pong.py --headless --seeds 100 --frames 10000
```

`--frames` e `--seeds` richiedono interi positivi. I seed della simulazione
headless vanno da zero a `--seeds - 1`; `--frames` indica i frame per seed.
Il risultato riporta respinte, palle mancate e punteggio aggregato: zero palle
mancate misura la capacità di difendere nelle condizioni provate, non quella di
battere ogni avversario. Il comando termina anche se vengono registrati errori
di gioco: lo stress test restituisce statistiche, senza imporre un punteggio minimo.

Il workflow `.github/workflows/tests.yml` esegue i test su Windows e Linux quando
la repository viene ospitata su GitHub. Il rendering nei test usa il driver SDL
`dummy`, quindi non richiede uno schermo.

## Come funziona

La rete `PongNetV5` ha struttura **2 → 8 → 2 → 1**, con 45 parametri tra pesi e
bias. Riceve la quota della racchetta e una previsione verticale grezza:

1. Python calcola il tempo all'impatto e `y_pred = ball_y + ball_vy * t`.
2. Le attivazioni ReLU del primo strato rappresentano i segmenti necessari a
   ripiegare la traiettoria dopo i rimbalzi, tramite una funzione triangolare
   chiamata *fold*.
3. Il secondo strato combina il fold, la posizione della racchetta e una zona
   morta di quattro pixel.
4. Lo strato finale produce il movimento: positivo verso il basso, negativo
   verso l'alto. La simulazione ne limita la velocità.

Il fold restituito per il grafico è diagnostico: il movimento viene calcolato
attraverso le matrici `W1`, `W2`, `W3`. Il centro della palla rimbalza tra
`y=16` e `y=634`; la rete usa gli stessi limiti e considera il raggio di 9 pixel.

La cinematica precedente alla rete resta scritta in Python. Inoltre la rete
implementa il fold soltanto su un intervallo finito: la previsione grezza viene
limitata a `[-2 * SPAN + 1, 2.9 * SPAN]`, con `SPAN=618`. Per traiettorie fuori
da questo intervallo, il limite può rendere imprecisa la previsione. La demo
conserva la grafica originale: il pannello del grafico copre parte del campo
e può essere nascosto con `G`.

## File

```text
pong.py                     Gioco, rete, simulazione e CLI
requirements.txt            Dipendenze con versioni bloccate
tests/test_pong.py           Test della rete, fisica e CLI
tests/test_app.py            Prova del rendering e dell'uscita
.github/workflows/tests.yml  Verifiche automatiche Windows/Linux
docs/superpowers/            Decisioni e piano della migrazione
```

Il precedente `pongNN.py` era un generatore che scriveva in `/mnt/data`, non
un punto di avvio del gioco. È stato sostituito da `pong.py`; l'originale resta
recuperabile nel primo commit della repository.
