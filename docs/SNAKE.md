# Snake: rete completa con scorciatoie

La rete predefinita osserva testa, mela, coda e corpo e sceglie le mosse con
strati affini e ReLU. Tutti i pesi sono costruiti analiticamente: nessun training,
reinforcement learning, ricerca di percorsi durante la partita o correzione
esterna dell'azione. La [baseline da 150 parametri](SNAKE_BASELINE.md) resta
disponibile con `python snake.py --policy cycle`.

## Osservazione

La griglia 18×14 ha N=252 celle. L'ingresso contiene quattro mappe binarie
piatte in ordine di riga, per **1008 valori**: testa, mela, coda e corpo completo.
Le prime tre hanno una sola cella attiva; l'ultima include testa e coda.

`encode_state(body, food)` controlla la validità e riempie le mappe, senza
calcolare azioni ammesse. `forward_observation(vector)` permette di esaminare
l'inferenza direttamente. Tutte le scelte restano nei coefficienti dei layer.

## Strategia e condizioni

Il ciclo della baseline fornisce un ordine alle celle. Le scorciatoie conservano
l'ordine ciclico coda→corpo→testa e non superano la mela. Il concetto di preservare
l'ordine durante le scorciatoie è descritto nell'[articolo originale di John
Tapsell](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/);
qui viene realizzato con una costruzione neurale specifica del progetto.

La rete penalizza i vicini fuori dal campo, nel corpo esclusa la coda mobile,
oltre la mela o oltre la coda. Fra gli altri sceglie il maggior avanzamento.
Quando il corpo occupa **almeno metà campo**, i pesi consentono soltanto il
successore immediato del ciclo: questo ricompatta il corpo ed evita buchi interni
costosi da raggiungere.

La proprietà di sicurezza dipende da:

- Corpo iniziale ordinato sul ciclo; avanzare prima della coda e rimuovere la
  vecchia coda conserva quest'ordine.
- Possibilità di entrare nella vecchia coda perché si libera nello stesso passo.
  La mela non può occuparla, quindi questo caso non comporta crescita.
- Disponibilità del successore del ciclo. Ogni mossa riduce la distanza dalla
  mela finché viene raggiunta, quindi non si crea un ciclo che la ignora.

Le condizioni sono campo vuoto, una mela alla volta, crescita immediata di un
segmento e stato iniziale coerente. Pesi modificati o corpi arbitrari non
conservano automaticamente queste proprietà. Non si rivendica il percorso
globalmente ottimo.

## Regole dentro le matrici

Architettura: **1008 → 955 → 28 → 11 → 20 → 4**. I primi quattro strati usano
ReLU; l'ultimo produce logits UP, RIGHT, DOWN, LEFT.

Il primo strato contiene un gate per ciascun arco adiacente p→q:

```text
occupato(p→q) = ReLU(testa[p] + corpo[q] - coda[q] - 1)
```

Altri neuroni ricavano sei differenze di rango traslate di N (quattro vicini,
mela, coda), indicatori dei bordi e lunghezza L sommando la mappa del corpo.
La geometria e i ranghi sono coefficienti costruiti una volta, non un percorso
calcolato in Python a ogni mossa.

Secondo e terzo strato costruiscono la distanza positiva modulo N da una
differenza traslata z:

```text
n0 = ReLU(N + 1 - z)
n1 = ReLU(N - z)
d = ReLU(z - N + N*n0 - N*n1)

T = N/2
allow = ReLU(ReLU(T-L) - ReLU(T-L-1))
```

Sulle osservazioni discrete la formula è esatta. Ranghi uguali danno distanza N,
utile per un corpo di lunghezza 1. La seconda formula consente scorciatoie
soltanto prima di metà campo.

Nel quarto strato, con f distanza dalla mela, t dalla coda e u indicatore
di occupazione/bordo:

```text
vf = ReLU(d - f)
vt = ReLU(d - t)
vd = ReLU(d - 1 - N*allow)
logit = d - (N+1)*(vf + vt + vd + u)
```

Una violazione intera porta il logit sotto ogni mossa ammessa. La simulazione
esegue direttamente l'argmax. Un test modifica il readout per scegliere il collo
occupato e verifica che il gioco registri la collisione: nessun filtro ripara
la scelta.

## Conteggio e rappresentazione

| Strato | Dimensioni | Pesi memorizzati | Bias |
| --- | --- | ---: | ---: |
| 1 | 1008→955 | 5100 | 955 |
| 2 | 955→28 | 968 | 28 |
| 3 | 28→11 | 28 | 11 |
| 4 | 11→20 | 32 | 20 |
| 5 | 20→4 | 20 | 4 |

Totale: **7.166 coefficienti memorizzati**, inclusi tutti i bias e gli zeri
espliciti. Le matrici dense equivalenti avrebbero **991.006 parametri**.
Ogni connessione memorizza indice sorgente, indice destinazione e coefficiente;
gli zeri impliciti non occupano matrici dense. L'inferenza rimane una somma
pesata, bias e ReLU, non una tabella di azioni.

La rete è più grande e costa più CPU della baseline. L'obiettivo è ridurre le
mosse del gioco; i report includono anche il tempo di esecuzione, comprensivo
di codifica e simulazione. I tempi dipendono dal carico della macchina.

## Uso e pesi

```powershell
.\.venv\Scripts\python.exe snake.py
.\.venv\Scripts\python.exe snake.py --test
.\.venv\Scripts\python.exe snake.py --compare --seeds 100 --json
.\.venv\Scripts\python.exe snake.py --compare --seed 100 --seeds 100 --json
.\.venv\Scripts\python.exe snake.py --weights weights/snake_shortcuts.json
.\.venv\Scripts\python.exe snake.py --policy cycle
```

- `shortcut_network.py`: costruzione, codifica e inferenza della rete completa.
- `weights/snake_shortcuts.json`: cinque matrici sparse, formato versione 2.
- `weights/snake_classic.json`: baseline originale, formato versione 1.
- `create_network()` e `load_network()`: selezione e caricamento delle politiche.

Il loader controlla dimensioni, indici e valori finiti e usa esattamente i
coefficienti caricati: non li rigenera. G mostra il ciclo di riferimento,
non il futuro percorso della politica.

## Valutazione

Il confronto usa gli stessi seed e regole. Le mele successive possono cadere
in celle diverse perché vengono estratte fra quelle libere. I seed 0–9 sono
stati usati durante la progettazione della regola; i seed 100–199 sono un gruppo
separato di verifica. Nessun aggiustamento dei pesi è basato su quel gruppo.
I risultati riproducibili sono in `docs/benchmarks/`.

Sui seed 0–99: baseline **15.465,24** mosse medie; rete completa **8.369,72**,
riduzione **45,88%**, entrambe con 100 completamenti e zero collisioni/timeout.
Questo dimostra il miglioramento in quelle condizioni, senza rivendicare una
nuova scoperta scientifica o superiorità su tutti i metodi per Snake.

Sui seed separati **100–199**: **15.427,51 → 8.380,99** mosse medie,
riduzione **45,68%**, ancora 100 completamenti su 100 per entrambe le reti.
Protocollo e limiti sono nel [riepilogo dei benchmark](benchmarks/README.md).
