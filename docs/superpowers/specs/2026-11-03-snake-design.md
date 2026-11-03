# Snake con pesi costruiti analiticamente

## Richiesta e scelta confermata

Aggiungere Snake con la grafica di https://snake-game.io/ e pesi costruiti
dall'assistente senza training. L'utente ha scelto Python/Pygame nella repository
attuale. Il riferimento è il gioco Classic, tema Sunset, serpente blu e mela rossa;
non il portale web circostante. Pong rimane disponibile.

## Riferimenti visivi

Acquisizioni in `output/playwright/snake-source-{desktop,settings,mobile}.png`.
Vista desktop del gioco: 908 x 801 px, barra alta 70 px; campo 18 x 14 celle
da 48 px, origine (22, 100). Colori campionati: cornice (183,80,48), barra
(100,77,31), scacchiera (251,193,78)/(252,178,79). Font Nunito Bold 22 px.
Mela e icone sono sprite pubblici del riferimento; il serpente è disegnato nel
canvas del sito e viene riprodotto in Pygame con lo stesso aspetto arrotondato.

## Decisioni implementative

- `snake.py`: CLI; package `snake_game` con rete, simulazione e app.
- Strategia esplicita: ciclo hamiltoniano attraverso tutte le celle. È una
  strategia conservativa, non una ricerca del percorso più corto verso la mela.
- Per 18 x 14, rete ReLU **2 -> 18 -> 4 -> 4**, 150 pesi/bias. Input: x,y
  della testa. Output: logits su/ destra/ giù/ sinistra. La mela non è un input.
- Prima layer: hinge ReLU per parità della riga e soglie dei bordi; secondo:
  gate per le svolte; ultimo: logits e priorità. Nessuna scelta della direzione
  esterna ai pesi, salvo l'argmax dell'uscita.
- Giro: righe pari verso sinistra, righe dispari verso destra; ultima colonna
  riservata al ritorno verso l'alto. Numero di righe pari.
- Vittoria locale quando tutte le celle sono occupate; questo stato terminale
  esplicito serve a misurare la copertura completa del controllore.
- Avvio automatico, pausa, riavvio, velocità, suono, schermo intero e aiuto.
  Impostazioni limitate alla demo Classic (velocità e visualizzazione percorso).
- Pesi esportabili e caricabili da JSON; copia standard nella repository.
- CLI headless riproducibile con risultati per seed; test di legalità dell'intero
  ciclo, crescita, collisioni, coda mobile, vittoria, caricamento pesi e UI.
- Asset locali con elenco fonti; nessun accesso alla rete durante il gioco.

## Alternative

Una strategia greedy verso la mela richiede controlli aggiuntivi contro le trappole
del corpo; una rete allenata contraddice la richiesta. Il ciclo codificato in una
piccola rete dà una proprietà verificabile su ogni cella e mantiene chiaro il
ruolo effettivo dei pesi. Nessuna garanzia è estesa a modalità con ostacoli.
