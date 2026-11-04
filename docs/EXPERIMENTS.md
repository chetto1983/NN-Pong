# Esperimenti sulle reti analitiche

Cinque esperimenti per capire che cosa dimostrano davvero le reti di questo
progetto. Codice in [`experiments/`](../experiments), risultati grezzi in JSON in
[`docs/experiments/`](experiments). Riproduzione:

```bash
python -m experiments.e1_pong_folds        # ~5 min
python -m experiments.e2_trained           # ~15 min
python -m experiments.e3_noise             # ~10 min
python -m experiments.e4_snake_references  # ~10 min
python -m experiments.e5_timing            # ~1 min, a CPU scarica
```

Tutti i numeri sotto vengono da questi file. NumPy, nessun framework di deep
learning; Python 3.13, 4 core.

## Sintesi

| Domanda | Risposta |
| --- | --- |
| Il test Pong "zero palle mancate" misura la rete? | **No.** A velocità normale non manca nessuna palla nemmeno una racchetta che insegue `ball_y` senza alcuna previsione. Il test diventa informativo solo da 2× la velocità di servizio. |
| Quanto costa togliere a Python la riduzione dei rimbalzi? | **5 parametri per ogni riflessione.** Con 55 parametri e intercetto grezzo i risultati coincidono con la versione a 15 parametri + modulo Python in tutti i regimi provati. |
| Una rete addestrata fa altrettanto? | **Pong sì, Snake no.** Pong: 33 parametri addestrati bastano in partita, ma non sono esatti e l'architettura analitica 2-2-2-1 non viene trovata dalla discesa del gradiente. Snake: 95% di accuratezza sugli stati, **0 vittorie su 60 partite**. |
| I pesi analitici sono robusti? | **Dipende dalla costruzione.** Snake con scorciatoie si rompe con rumore relativo 10⁻⁴ (non regge bfloat16); il ciclo regge 10⁻³; Pong regge 10⁻². Con rumore forte la rete Pong addestrata è più robusta di quella analitica. |
| Quanto è buona la politica Snake? | Vince sempre, su ogni dimensione provata (6×4 – 24×18). Un greedy sicuro fa meno mosse quando vince, ma vince 7 volte su 30. Il guadagno sul ciclo viene tutto dalla prima metà della partita. |
| Quanto costa la forma neurale? | Per una decisione singola: Pong 11×, Snake 21×, ciclo 110× più lenta della stessa regola in Python. In batch NumPy, Pong è invece 3× più veloce. |

## E1 — Rimbalzi dentro o fuori dalla rete (Pong)

[`e1_pong_folds.py`](../experiments/e1_pong_folds.py) ·
[risultati](experiments/e1_pong_folds.json)

La rete distribuita riceve l'intercetto già ridotto modulo `2*SPAN` da Python.
Qui una famiglia di reti esatte ripiega da sola l'intercetto grezzo: ogni
riflessione in più richiede **una unità hinge, cioè 5 parametri**
(`parametri = 5·H + 5`). L'errore massimo sul dominio coperto è ≤ 6·10⁻¹².

Distribuzione delle riflessioni negli intercetti reali (50 seed × 10.000 passi,
251.634 passi di controllo con palla in arrivo): 0 riflessioni 79,9%, 1 14,8%,
2 1,5%, oltre il 3,8%. La coda arriva a ±28.000 px, cioè 45 riflessioni (vedi
"Difetto trovato" sotto).

| Parametri | Copertura degli intercetti reali |
| ---: | ---: |
| 15 | 89,5% |
| 30 | 96,3% |
| 55 | 97,4% |

Fuori dal dominio l'input viene limitato al bordo e la rete sbaglia, ma la
previsione si corregge mentre la palla si avvicina. Mancate per velocità di
servizio (30 seed × 5.000 passi):

| Velocità | Solo inseguimento, nessuna previsione | 15 param., intercetto grezzo | 55 param., intercetto grezzo | 15 param. + modulo Python (distribuita) |
| ---: | ---: | ---: | ---: | ---: |
| 1× | 0 / 448 | 0 / 478 | 0 / 478 | 0 / 478 |
| 2× | 4 / 967 | 0 / 953 | 0 / 953 | 0 / 953 |
| 3× | 11 / 1.458 | 1 / 1.458 | 0 / 1.454 | 0 / 1.454 |
| 4× | 63 / 1.889 | 7 / 1.938 | 0 / 1.830 | 0 / 1.830 |
| 5× | 248 / 2.224 | 41 / 2.335 | 30 / 2.093 | 30 / 2.085 |

(mancate / palle arrivate). Il massimo regolabile in gioco è 2,5×.

**Conclusioni.** A velocità normale il benchmark Pong non distingue nulla. La
previsione conta da 2×. In partita, il modulo Python è sostituibile da 40
parametri in più con risultati identici (la copertura resta 97,4%: non è
un'equivalenza su ogni input). A 5× falliscono tutti: è il limite fisico della velocità
della racchetta, non della rete.

### Difetto trovato

Un urto sulla calotta arrotondata di una racchetta può lasciare la palla quasi
verticale (seed 28: `|vx|` = 6,5 px/s, ~150 s simulati per attraversare il
campo). Manca una velocità orizzontale minima. Non è stato corretto qui perché
cambierebbe tutti i benchmark; va deciso a parte.

## E2 — Reti addestrate contro pesi analitici

[`e2_trained.py`](../experiments/e2_trained.py) ·
[risultati](experiments/e2_trained.json)

Imitazione supervisionata della politica analitica (behaviour cloning), Adam,
etichette esatte e illimitate: il caso più favorevole all'addestramento.

**Pong** (dominio ridotto, 3 inizializzazioni, 20.000 passi; partite 10 seed ×
5.000 passi):

| Architettura | Parametri | Errore max | Direzione sbagliata | Mancate 1× | Mancate 3× |
| --- | ---: | ---: | ---: | ---: | ---: |
| analitica 2-2-2-1 | 15 | 0 | 0% | 0 | 0 |
| addestrata 2-2-2-1 | 15 | 316–629 | 24–50% | 114–136 | 319–332 |
| addestrata 2-8-1 | 33 | 8–13 | 0% | 0 | 0 |
| addestrata 2-32-1 | 129 | 7 | 0% | 0 | 0 |
| addestrata 2-32-32-1 | 1.185 | 6–8 | 0% | 0 | 0 |

La soluzione esatta a 15 parametri esiste, ma la discesa del gradiente non la
trova (minimo locale con unità morte). Con poco margine in più basta
un'approssimazione da ~7 px, che in partita non si distingue.

**Snake** (109.903 stati da 40 partite per l'addestramento, 27.868 da 10 partite
diverse per il test, 15 epoche; 20 partite su seed nuovi, limite 30.000 mosse):

| Architettura | Parametri | Acc. addestramento | Acc. test | Vittorie | Mele medie |
| --- | ---: | ---: | ---: | ---: | ---: |
| analitica sparsa | 7.166 (991.006 dense) | 100% | 100% | 20/20 | 248 |
| 1008-64-4 | 64.836 | 95,9% | 94,1% | 0/20 | 2,1 |
| 1008-256-4 | 259.332 | 98,3% | 94,7% | 0/20 | 0,0 |
| 1008-256-256-4 | 325.124 | 100% | 94,7% | 0/20 | 2,1 |

Il 5% di errori basta a portare il serpente in stati mai visti, dove ripete un
giro senza mangiare (19–20 partite su 20 finiscono per limite di mosse). È il
classico accumulo d'errore del behaviour cloning.

**Limiti.** È provato solo il metodo di addestramento più diretto. DAgger,
architetture convoluzionali o RL potrebbero fare molto meglio; questo
esperimento dice che *imitare* la politica non basta, non che una rete
addestrata non possa giocare a Snake.

## E3 — Robustezza al rumore e alla precisione

[`e3_noise.py`](../experiments/e3_noise.py) ·
[risultati](experiments/e3_noise.json)

Ogni coefficiente memorizzato diventa `w·(1 + σ·ε)`, ε ~ N(0,1): gli zeri
strutturali restano zero e σ si legge come precisione relativa. Pong è giocato
a 3× (vedi E1), Snake su 5 partite per livello.

| σ | Pong analitica: errore max / mancate | Pong addestrata 1.185: errore max / mancate | Snake scorciatoie: decisioni uguali / vittorie | Snake ciclo: decisioni uguali / vittorie |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 / 0 | 7,8 / 0 | 100% / 5 | 100% / 5 |
| 10⁻⁶ | 0,007 / 0 | 7,8 / 0 | 100% / 5 | 100% / 5 |
| 10⁻⁵ | 0,07 / 0 | 7,8 / 0 | 99,5% / 5 | 100% / 5 |
| 10⁻⁴ | 0,7 / 0 | 7,8 / 0 | 85,6% / **0** | 100% / 5 |
| 10⁻³ | 7,1 / 0 | 7,9 / 0 | 85,2% / 0 | 100% / 5 |
| 10⁻² | 71 / 0 | 13 / 0 | 48% / 0 | 70% / **0** |
| 10⁻¹ | 700 / **405** | 92 / 0 | 23% / 0 | 36% / 0 |

**Previsioni a priori, poi verificate.**
- Snake con scorciatoie: le differenze di rango (fino a 251) passano per un
  gradino `ReLU(N+1−Z) − ReLU(N−Z)` moltiplicato per N = 252, quindi un errore
  δ diventa ~252·δ. Stima della soglia ≈ 5·10⁻⁶; osservata fra 10⁻⁵ e 10⁻⁴.
- Ciclo: margini dell'ordine di 1 su somme di circa 180 → soglia ≈ 3·10⁻³;
  osservata fra 10⁻³ e 10⁻².
- Pong: errore ≈ 7.000·σ px, misurato esattamente lineare. Conta solo quando
  supera la zona morta e la tolleranza della racchetta.

**Precisione numerica** (pesi e uscita di ogni strato arrotondati):

| Formato | Pong: errore max / mancate 3× | Snake scorciatoie: decisioni uguali / vittorie |
| --- | ---: | ---: |
| float32 | 0,0001 / 0 | 100% / 5 |
| float16 | 1,07 / 0 | 100% / 5 |
| bfloat16 | 17 / 0 | 99,2% / **0** |

I coefficienti di Snake sono interi ≤ 253, rappresentati esattamente in
float16. bfloat16 ha solo 8 bit di mantissa e li arrotonda; i valori penalizzati
in float16 vanno in overflow a −∞, ma l'argmax resta corretto.

**Conclusioni.** I pesi analitici non sono robusti per natura: la robustezza
dipende dai margini che la costruzione lascia. Snake con scorciatoie usa
coefficienti grandi moltiplicati fra loro e non ha margine; il ciclo ne ha
cento volte di più. Contro rumore forte la rete addestrata, più "liscia", regge
meglio. Uno 0,8% di decisioni diverse basta a impedire ogni vittoria: in Snake
l'errore non resta locale.

## E4 — Snake contro riferimenti esterni

[`e4_snake_references.py`](../experiments/e4_snake_references.py) ·
[risultati](experiments/e4_snake_references.json)

Campo 18×14, seed 0–29. Il *greedy sicuro* va verso la mela per il cammino più
breve solo se dopo può ancora raggiungere la coda, altrimenti segue la coda
(euristica standard, nessuna garanzia; limite 30.000 mosse).

| Politica | Vittorie | Mosse medie se vince | Mosse/mela prima metà | Mosse/mela seconda metà | Mosse / limite di Manhattan |
| --- | ---: | ---: | ---: | ---: | ---: |
| ciclo (rete 150 param.) | 30/30 | 15.402 | 93,9 | 31,4 | 6,4 |
| scorciatoie (rete 7.166 param.) | 30/30 | 8.226 | 34,6 | 31,8 | 3,5 |
| greedy sicuro (Python) | 7/30 | 9.382 | 20,8 | 54,4 | — |

Il greedy non muore mai, ma in 23 partite su 30 si blocca in un giro che segue
la coda (almeno 239 mele per partita, 245,5 in media). Il rapporto con il limite di Manhattan misura le
deviazioni sulla sequenza di mele effettivamente vista, non la distanza
dall'ottimo: un'altra politica vedrebbe mele diverse.

**La seconda metà non migliora nulla.** Da metà campo occupato la rete torna
al ciclo (31,8 mosse/mela contro 31,4 della baseline). Tutto il −46% viene dalla
prima metà. È qui che si potrebbe guadagnare ancora, se esistesse una regola di
scorciatoie sicura con il corpo lungo.

**Altre dimensioni** (10 seed ciascuna; la rete è ricompilata per ogni campo):

| Campo | Celle | Coefficienti memorizzati | Dense equivalenti | Vittorie scorciatoie / ciclo | Riduzione mosse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6×4 | 24 | 774 | 11.546 | 10 / 10 | −19,7% |
| 8×6 | 48 | 1.430 | 39.346 | 10 / 10 | −30,5% |
| 10×8 | 80 | 2.318 | 103.626 | 10 / 10 | −35,0% |
| 12×10 | 120 | 3.438 | 228.194 | 10 / 10 | −41,0% |
| 14×12 | 168 | 4.790 | 443.002 | 10 / 10 | −43,4% |
| 18×14 | 252 | 7.166 | 991.006 | 10 / 10 | −47,3% |
| 24×18 | 432 | 12.286 | 2.908.506 | 10 / 10 | −47,6% |

I coefficienti memorizzati crescono linearmente con le celle, quelli densi in
modo quadratico. La riduzione tende a circa −48%.

## E5 — Costo della forma neurale

[`e5_timing.py`](../experiments/e5_timing.py) ·
[risultati](experiments/e5_timing.json)

Stessa decisione, prima come rete e poi come codice Python. Prima della misura
la regola Python di Snake ([`policies.py`](../experiments/policies.py)) è stata
verificata identica ai logit della rete su 1.546 stati (e su 23.160 durante lo
sviluppo).

| Decisione | Rete | Python | Rapporto |
| --- | ---: | ---: | ---: |
| Pong, singola | 5,4 µs | 0,48 µs | 11× |
| Pong, batch NumPy (per elemento) | 0,006 µs | 0,020 µs | **0,3×** |
| Snake scorciatoie, singola | 159 µs (116 di codifica e validazione) | 7,6 µs | 21× |
| Snake ciclo, singola | 5,7 µs | 0,05 µs (tabella) | 110× |

In batch la rete Pong è più veloce perché sono tre prodotti matrice-vettore,
contro modulo e `where` in NumPy. Per decisioni singole la forma neurale costa
da 10 a 100 volte di più. In Snake pesa soprattutto la costruzione delle
quattro mappe da 1.008 valori.

## Che cosa cambia nella lettura del progetto

1. **Pong non è una prova della rete.** Il benchmark va reso informativo (2× o
   più) e il difetto delle traiettorie quasi verticali va corretto.
2. **"Tutto nei pesi" ha un prezzo misurabile.** In Pong 5 parametri per
   riflessione, in Snake una forte fragilità numerica.
3. **Il valore dei pesi analitici è la garanzia, non l'efficienza.** Esattezza
   e completamento su ogni seed e dimensione provati, dove l'imitazione
   addestrata fallisce del tutto. Il costo è l'assenza di margine numerico.
4. **Possibile miglioramento di Snake.** Scorciatoie sicure anche con il corpo
   oltre metà campo: oggi quella metà della partita non guadagna nulla.
