# NN Pong e Snake

Due giochi automatici controllati da reti con **pesi costruiti analiticamente**,
senza addestramento: Pong e Snake Classic. Snake riprende la grafica del gioco
su [snake-game.io](https://snake-game.io/) e usa una rete che osserva mela e corpo
per scegliere scorciatoie; la baseline originale da 150 parametri resta disponibile.

Pong automatico in Python e Pygame: la racchetta turchese è controllata da una
piccola rete ReLU con pesi costruiti analiticamente, senza addestramento.
La racchetta corallo usa una strategia deterministica di confronto.

L'idea nasce da [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
di Rikiya Takehi, Jayden Teoh, Alex Zhang e Omar Khattab. Questo progetto è una
dimostrazione indipendente ispirata al principio di esprimere un algoritmo nei
pesi di una rete; non riproduce i benchmark dell'articolo.

## Quanti parametri hanno le reti

Il conteggio comprende pesi e bias. La rete completa di Snake usa matrici
sparse: i coefficienti memorizzati vanno distinti dalle dimensioni delle
matrici dense equivalenti.

| Rete | Architettura | Pesi memorizzati | Bias | Totale memorizzato | Equivalente denso |
| --- | --- | ---: | ---: | ---: | ---: |
| Pong | 2 → 8 → 2 → 1 | 34 | 11 | **45** | 45 |
| Snake originale, ciclo | 2 → 18 → 4 → 4 | 124 | 26 | **150** | 150 |
| Snake completa, scorciatoie | 1008 → 955 → 28 → 11 → 20 → 4 | 6.148 | 1.018 | **7.166** | 991.006 |

I **7.166** coefficienti della rete completa includono tutti i bias e gli zeri
esplicitamente memorizzati. **991.006** è il conteggio che si avrebbe allocando
tutte le connessioni, comprese quelle implicitamente nulle; non è il numero
di coefficienti salvati nel modello sparse. Gli indici delle connessioni e
gli altri metadati occupano memoria aggiuntiva, ma non sono inclusi nel
conteggio di pesi e bias.

Tutti i pesi delle reti di gioco sono fissi e costruiti analiticamente, senza
training o RL. La variante equivalente di Snake da **129 parametri** è stata
provata soltanto in memoria: non è una delle reti distribuite né la baseline
usata nei benchmark.

I pesi Snake sono leggibili in [snake_classic.json](weights/snake_classic.json)
e [snake_shortcuts.json](weights/snake_shortcuts.json). La costruzione dei pesi
Pong è in [pong_game/network.py](pong_game/network.py).

## Requisiti e avvio

Python **3.12 o superiore**. Versioni delle dipendenze bloccate in
`requirements.txt`; ambiente locale verificato con Python 3.13 su Windows.

Da PowerShell, nella cartella del progetto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pong.py

# Oppure Snake
.\.venv\Scripts\python.exe snake.py
```

Non serve attivare l'ambiente virtuale. Se `.venv` è già stato preparato, basta
l'ultimo comando. Su Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pong.py
```

Entrambe le racchette sono automatiche. Il campo e il punteggio sono sempre
visibili; uscita della rete, previsione e statistiche si consultano nel pannello
di diagnostica, separato dall'area di gioco. La finestra è ridimensionabile.

| Tasto | Azione |
| --- | --- |
| `Esc` | Esci |
| `Spazio` | Pausa / riprendi |
| `R` | Ricomincia la partita |
| `G` | Mostra / nascondi la diagnostica fuori dal campo |
| `+` oppure `=` | Aumenta la velocità della palla |
| `-` | Riduci la velocità della palla |
| `F` | Schermo intero / finestra |

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
headless vanno da zero a `--seeds - 1`; `--frames` indica passi di 1/60 di
secondo per seed, indipendentemente dalla velocità del computer.
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

1. Python calcola il tempo all'impatto e `y_pred = ball_y + ball_vy * t`, poi
   riduce la coordinata modulo il periodo di riflessione per tenerla nel
   dominio della rete senza cambiare il punto di arrivo.
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

La cinematica precedente alla rete resta scritta in Python. Il vecchio limite
che tagliava le previsioni molto lontane è stato rimosso; la periodicità consente
di gestire anche traiettorie con molti rimbalzi.

## Fisica di Pong

La palla si muove a velocità costante fra gli urti. Le collisioni sono calcolate
al tempo di contatto fra il cerchio della palla e le racchette a capsula, anche
quando lo spostamento supera lo spessore della racchetta. Sulle pareti cambia
la componente normale della velocità; sulle racchette intervengono anche il
movimento della racchetta e l'attrito tangenziale con spin. Non c'è più un
aumento automatico della velocità a ogni respinta.

L'app usa passi fisici di 1/120 di secondo e interpolazione per il rendering.
Pareti e racchette disegnate corrispondono alla geometria simulata. È un modello
di urti 2D senza gravità, non una simulazione tridimensionale del ping-pong.
Dettagli e limiti in [docs/PONG_PHYSICS.md](docs/PONG_PHYSICS.md).

![Nuova schermata Pong](docs/images/pong.png)

Verifica locale della revisione Pong: **92 test superati** al momento della revisione;
stress Pong di 100 seed × 10.000 passi, 3.019 respinte, zero palle mancate
dalla rete e punteggio aggregato 38–0. Questi risultati riguardano l'avversario
e le condizioni inclusi nella demo.

## File

```text
pong.py                     CLI e compatibilità degli import
pong_game/                  Configurazione, rete, fisica e renderer
requirements.txt            Dipendenze con versioni bloccate
tests/test_pong.py           Test della rete, fisica e CLI
tests/test_pong_physics.py   Regressioni degli urti e della temporizzazione
tests/test_app.py            Rendering, comandi e ridimensionamento
.github/workflows/tests.yml  Verifiche automatiche Windows/Linux
docs/superpowers/            Decisioni e piano della migrazione
```

Il precedente `pongNN.py` era un generatore che scriveva in `/mnt/data`, non
un punto di avvio del gioco. È stato sostituito da `pong.py`; l'originale resta
recuperabile nel primo commit della repository.

## Snake Classic

```powershell
# Avvia la finestra Pygame
.\.venv\Scripts\python.exe snake.py

# Verifica ogni cella del ciclo e una partita completa
.\.venv\Scripts\python.exe snake.py --test

# Cento partite riproducibili, senza finestra
.\.venv\Scripts\python.exe snake.py --headless --seeds 100

# Confronta rete completa e baseline sugli stessi seed
.\.venv\Scripts\python.exe snake.py --compare --seeds 100

# Usa i pesi della rete completa inclusi nella repository
.\.venv\Scripts\python.exe snake.py --weights weights/snake_shortcuts.json

# Rete originale, senza scorciatoie
.\.venv\Scripts\python.exe snake.py --policy cycle

# Esporta una nuova copia dei pesi analitici
.\.venv\Scripts\python.exe snake.py --export-weights pesi.json
```

La finestra riprende **Classic, tema Sunset**: scacchiera arancione 18×14,
serpente blu, mela rossa, font e icone del riferimento. Le altre modalità del
sito non fanno parte di questa demo. Tutti gli asset sono locali e il gioco
funziona offline; fonti in [snake_assets/SOURCES.md](snake_assets/SOURCES.md).

| Controllo Snake | Azione |
| --- | --- |
| `Spazio` | Pausa / riprendi |
| `R` | Nuova partita con lo stesso seed |
| `+` / `-` | Velocità, da 1 a 120 mosse al secondo |
| `G` | Mostra / nascondi il ciclo di riferimento |
| `H` o icona `?` | Aiuto |
| Ingranaggio | Velocità e visualizzazione del percorso |
| `F` o icona schermo intero | Cambia visualizzazione |
| Altoparlante | Attiva / disattiva il suono |
| `Esc` | Chiudi il pannello aperto oppure esci |

Il serpente parte automaticamente. Le frecce possono riprendere una pausa,
ma la direzione rimane decisa dalla rete. Il record locale è salvato in
`~/.nn-pong/snake-record.json`; le simulazioni headless non lo modificano.

La nuova rete **1008 → 955 → 28 → 11 → 20 → 4** riceve quattro mappe binarie:
testa, mela, coda e corpo. Nei pesi sono codificati ostacoli, distanze sul ciclo
e scelta delle scorciatoie. Da metà campo occupato ritorna al ciclo per
ricompattare il corpo. In partita esegue soltanto strati affini, ReLU e argmax,
senza training, ricerca di percorsi o sostituzione esterna dell'azione.

Contiene **7.166 coefficienti memorizzati** in matrici sparse (991.006 nella
rappresentazione densa equivalente). È più grande e costa più CPU della baseline:
il guadagno riguarda le mosse necessarie a completare il gioco. La proprietà di
sicurezza riguarda i pesi costruiti, il campo vuoto e il corpo inizializzato
sul ciclo; modificare pesi, corpo o ostacoli può invalidarla.

La demo dichiara vittoria quando occupa tutte le 252 celle (248 mele dopo i
quattro segmenti iniziali). Questo stato terminale esplicito consente di misurare
il completamento del controllore. `--headless` restituisce codice 1 se una partita
muore o supera `--max-steps` (default 100.000); `--json` produce statistiche
leggibili da altri programmi. `--seed` imposta il primo seed della serie.

La costruzione matematica dei pesi è descritta in [docs/SNAKE.md](docs/SNAKE.md).
I test Snake sono inclusi nel comando `python -m unittest discover -s tests -v`.

Confronto sui seed **0–99**: **100 completamenti su 100** per entrambe le reti,
zero collisioni e 248 mele per partita. Media: **15.465,24 → 8.369,72 mosse**,
una riduzione del **45,88%**. Report JSON in [docs/benchmarks](docs/benchmarks/).

Sui seed separati **100–199**, il risultato è **15.427,51 → 8.380,99 mosse**
(**−45,68%**), con altri 100 completamenti su 100 e zero collisioni.
La suite completa passa **115 test**. [Protocollo e limiti del confronto](docs/benchmarks/README.md).

![Snake Classic in Pygame](docs/images/snake.png)
