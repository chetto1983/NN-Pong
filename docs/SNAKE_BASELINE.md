# Snake: un programma scritto nei pesi

Questa pagina descrive la baseline da 150 parametri, selezionabile con
`python snake.py --policy cycle`. La rete predefinita con scorciatoie è descritta
in [SNAKE.md](SNAKE.md).

L'idea è coerente con il [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/):
costruire le matrici che realizzano un comportamento, senza ottimizzatore o training.
Questo controllore è un esperimento indipendente, non una replica dei benchmark
dell'articolo.

## Strategia

Su un campo rettangolare con un numero pari di righe, la rete segue un ciclo
hamiltoniano: ogni cella viene visitata una volta prima di tornare all'inizio.
Le righe pari vengono percorse verso sinistra, quelle dispari verso destra.
L'ultima colonna è riservata al ritorno verso l'alto.

Il corpo parte con quattro segmenti consecutivi sul ciclo. Ogni mossa procede
alla cella successiva; la crescita occupa un'altra cella del ciclo senza tagliare
il corpo. Finché c'è una cella libera, ogni mela generata su una cella libera
verrà raggiunta. Il programma termina con vittoria quando il campo è pieno.

La strategia privilegia una proprietà verificabile rispetto al numero minimo
di mosse per mela. Non usa la posizione della mela come input e non cerca scorciatoie.

## Rete 2 → 18 → 4 → 4

Input: coordinate intere `(x, y)` della testa su un campo `W=18`, `H=14`.
Tutte le coordinate vengono validate prima dell'inferenza. `ReLU(t)=max(0,t)`.

Il primo strato produce 18 attivazioni:

- `a[k] = ReLU(y-k)`, per `k=0…H-2` (13 attivazioni).
- `R = ReLU(x-(W-2))`, indicatore dell'ultima colonna.
- `L = ReLU(1-x)`, indicatore della prima colonna.
- `Q = ReLU(x-(W-3))`, usato per riconoscere la penultima colonna.
- `T = ReLU(1-y)`, indicatore della prima riga.
- `B = ReLU(y-(H-2))`, indicatore dell'ultima riga.

Nella combinazione lineare del secondo strato:

```text
O = a[0] + Σ(k=1…H-2) 2·(-1)^k·a[k]
P = Q - 2R
```

Sulle coordinate intere valide, `O` è 1 sulle righe dispari e 0 sulle pari;
`P` vale 1 soltanto sulla penultima colonna. Sono combinazioni dei neuroni del
primo strato, incluse nella matrice `W2`, non calcoli Python che scelgono la mossa.

Il secondo strato contiene quattro gate:

```text
u = ReLU(R - T)
d1 = ReLU(L - O)
d2 = ReLU(P + O - B - 1)
o = ReLU(O)
```

L'ultimo strato produce i logits nell'ordine **su, destra, giù, sinistra**:

```text
[3u, o, 3(d1+d2), 1-o]
```

Il valore 3 dà priorità alle svolte sui bordi rispetto al movimento lungo le
righe. L'azione è l'argmax dei quattro logits. L'inferenza usa soltanto prodotti
matrice-vettore, bias e ReLU; non contiene una funzione alternativa di guida.

| Tensore | Dimensioni | Parametri |
| --- | --- | --- |
| W1, b1 | 18×2, 18 | 54 |
| W2, b2 | 4×18, 4 | 76 |
| W3, b3 | 4×4, 4 | 20 |
| Totale | | **150** |

## Cosa resta fuori dalla rete

Python gestisce lo stato del gioco: movimento del corpo, comparsa casuale delle
mele, collisioni, punteggio, rendering e argmax. Non modifica i pesi in base ai
risultati e non sostituisce le azioni della rete con mosse ritenute migliori.

Il percorso mostrato con `G` è diagnostico, ricavato dalle uscite della rete;
non costituisce una tabella usata dal gioco per scegliere le mosse.

## File e riproducibilità

- `snake_game/network.py`: costruzione analitica e inferenza.
- `snake_game/simulation.py`: regole e stato, indipendenti da Pygame.
- `snake_game/app.py`: finestra, animazione, controlli e record.
- `snake.py`: CLI, verifica del ciclo, benchmark ed esportazione.
- `weights/snake_classic.json`: i 150 valori, leggibili e modificabili.

Il JSON include versione del formato, dimensioni della griglia e sei tensori.
Il caricamento verifica forme e valori finiti. La compatibilità del formato
non implica la sicurezza di una politica modificata: usare `--test` e
`--headless` per valutare nuovi pesi.

La rete supporta campi con almeno 6 colonne e almeno 4 righe, in numero pari.
Il numero di neuroni del primo strato è `H+4`. La finestra riproduce il campo
18×14 del riferimento; altri campi possono essere valutati tramite API o pesi
caricati in modalità headless. La geometria dello stato iniziale e l'assenza
di ostacoli sono parte delle condizioni della strategia.
