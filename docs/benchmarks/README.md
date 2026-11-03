# Confronto Snake — 3 novembre 2026

| Seed | Ciclo: mosse medie | Rete completa: mosse medie | Riduzione | Completamenti per politica |
| --- | ---: | ---: | ---: | ---: |
| 0–99 | 15.465,24 | 8.369,72 | 45,88% | 100/100 |
| 100–199, verifica separata | 15.427,51 | 8.380,99 | 45,68% | 100/100 |

Entrambe le politiche hanno raccolto 248 mele per partita, senza collisioni o
timeout nei due gruppi. La rete completa ha effettuato 62.852 mosse di scorciatoia
nel primo gruppo e 62.633 nel secondo.

## Riproduzione

```powershell
.\.venv\Scripts\python.exe snake.py --compare --seeds 100 --json
.\.venv\Scripts\python.exe snake.py --compare --seed 100 --seeds 100 --json
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Report integrali: [seed 0–99](snake-comparison-0-99.json) e
[seed 100–199](snake-holdout-100-199.json). La versione Python locale è 3.13;
NumPy e Pygame restano quelli bloccati in `requirements.txt`.

## Protocollo

- Campo 18×14, quattro segmenti iniziali, una mela alla volta, crescita immediata.
- La baseline è la rete originale da 150 parametri, con gli stessi pesi e
  lo stesso ciclo. La candidata usa 7.166 coefficienti memorizzati in matrici
  sparse, equivalenti a 991.006 parametri nella rappresentazione densa.
- Limite 100.000 mosse per partita; vittoria al riempimento del campo.
- Stessi seed e regole. I posti delle mele possono differire perché la lista
  delle celle libere dipende dal corpo; non è un confronto su mele collocate
  obbligatoriamente nelle stesse coordinate.
- I seed 0–9 sono stati usati nella progettazione: sono state confrontate
  regole di avanzamento, riserva di spazio e soglie di densità. Scelta una
  soglia di metà campo, i coefficienti sono stati costruiti direttamente.
  Non sono stati usati discesa del gradiente, aggiornamenti RL o fitting dei pesi.
- I seed 100–199 non sono stati usati per modificare la regola o i pesi.
- I test confrontano la rete con un oracolo distinto, verificano l'ordine del
  corpo e forzano pesi sbagliati per escludere un correttore esterno delle azioni.

## Cosa dimostra e cosa non dimostra

Il risultato mostra una riduzione riproducibile delle mosse rispetto alla
baseline ciclica in questo ambiente. Non dimostra ottimalità globale,
superiorità rispetto ad A*, RL o altri solver, né novità scientifica rispetto
alla letteratura. La progettazione ha usato il feedback delle simulazioni:
non è un esperimento zero-shot senza accesso all'ambiente.

La rete completa richiede più CPU: nei report, circa 191–192 secondi per
100 partite contro circa 24–25 secondi della baseline. Sono tempi indicativi,
comprendono codifica e simulazione e sono stati misurati mentre altre verifiche
erano attive. Il miglioramento riguarda il numero di mosse, non il tempo CPU.

La proprietà di sicurezza è condizionata allo stato iniziale ordinato sul
ciclo, alle regole descritte e ai pesi costruiti. Stati o pesi arbitrari non
sono automaticamente sicuri. Il caricamento verifica il formato, non certifica
la politica.

## Verifica della consegna

- 115 test della repository superati, inclusi Pong e Snake.
- Revisione indipendente: nessun rilievo sul codice esaminato; verificati
  inferenza sparse, ordine ciclico, coda mobile, loader e confronto incompleto.
- Windows nativo: 600 frame renderizzati, 100 mosse, quattro mele e 22
  scorciatoie; aiuto, pausa, ciclo di riferimento, riavvio e uscita verificati.
- Nessuna nuova dipendenza o modifica ai pesi della baseline.
