# Esperimenti sulle reti analitiche

Sei esperimenti per capire che cosa dimostrano davvero le reti di questo
progetto, e che cosa no. Codice in [`experiments/`](../experiments), risultati
grezzi in JSON in [`docs/experiments/`](experiments). Riproduzione:

```bash
python -m experiments.e1_pong_folds        # ~10 min
python -m experiments.e2_trained           # ~20 min
python -m experiments.e3_noise             # ~15 min
python -m experiments.e4_snake_references  # ~10 min
python -m experiments.e6_dagger            # ~30 min
python -m experiments.e5_timing            # ~1 min, da eseguire a CPU scarica
```

Tutti i numeri sotto vengono da questi file, con la fisica Pong che limita il
rimbalzo a 60° (vedi E1). NumPy, nessun framework di deep learning; Python 3.13,
4 core. I tempi di addestramento dipendono dal carico: quattro esperimenti
giravano in parallelo.

## Sintesi

| Domanda | Risposta |
| --- | --- |
| Il test Pong a velocità normale misura la rete? | **No.** Nemmeno una racchetta che insegue `ball_y` senza previsione manca una palla. Il benchmark informativo è a 3× (rete 0 mancate su 8.961, inseguimento 58 su 9.589). |
| Quanto costa togliere a Python la riduzione dei rimbalzi? | **5 parametri per riflessione.** Con 50 parametri e intercetto grezzo si copre il 100% degli intercetti osservati e i risultati coincidono con la versione distribuita. |
| Una rete addestrata fa altrettanto? | **Pong sì, Snake no.** Pong: bastano 33 parametri addestrati, ma la discesa del gradiente non trova la soluzione esatta a 15. Snake: imitazione semplice 0/60 vittorie; **DAgger** dopo 10 iterazioni sale da 2 a ~30 mele su 248, ancora **0/20 vittorie**. |
| I pesi analitici sono robusti? | **Dipende dalla costruzione.** Snake con scorciatoie si rompe con rumore relativo 10⁻⁴ e in bfloat16; il ciclo regge 10⁻³; Pong 10⁻². Con rumore forte la rete Pong addestrata è più robusta. |
| Quanto è buona la politica Snake? | Vince sempre, su tutti i campi provati (6×4 – 24×18). Un greedy sicuro vince 7 volte su 30. Il guadagno sul ciclo viene tutto dalla prima metà della partita. |
| Quanto costa la forma neurale? | Per decisione singola: Pong 13×, Snake 20×, ciclo 100× più lenta della stessa regola in Python. In batch NumPy Pong è invece 2× più veloce. |
| Questo dimostra che un LLM batte l'addestramento? | **No.** Vedi l'ultima sezione e il confronto con l'articolo. |

## E1 — Rimbalzi dentro o fuori dalla rete (Pong)

[`e1_pong_folds.py`](../experiments/e1_pong_folds.py) ·
[risultati](experiments/e1_pong_folds.json)

La rete distribuita riceve l'intercetto già ridotto modulo `2*SPAN` da Python.
Qui una famiglia di reti esatte ripiega da sola l'intercetto grezzo: ogni
riflessione in più richiede **una unità hinge, cioè 5 parametri**
(`parametri = 5·H + 5`). Errore massimo sul dominio coperto ≤ 6·10⁻¹².

Riflessioni negli intercetti reali (50 seed × 10.000 passi, 251.002 passi di
controllo con palla in arrivo): 0 riflessioni 81,8%, 1 16,3%, 2 1,9%, 3 0,02%.
Intercetti fra −1.538 e 2.176 px.

| Parametri | Copertura degli intercetti reali |
| ---: | ---: |
| 15 | 91,4% |
| 20 | 92,7% |
| 30 | 99,98% |
| 50 | 100% |

Fuori dal dominio l'input viene limitato al bordo e la rete sbaglia, ma la
previsione si corregge mentre la palla si avvicina. Mancate / palle arrivate
per velocità di servizio (30 seed × 5.000 passi):

| Velocità | Inseguimento, nessuna previsione | 15 param., intercetto grezzo | 55 param., intercetto grezzo | 15 param. + modulo Python (distribuita) |
| ---: | ---: | ---: | ---: | ---: |
| 1× | 0 / 460 | 0 / 455 | 0 / 455 | 0 / 455 |
| 2× | 5 / 952 | 0 / 954 | 0 / 954 | 0 / 954 |
| 3× | 10 / 1.456 | 3 / 1.433 | 0 / 1.422 | 0 / 1.422 |
| 4× | 75 / 1.865 | 8 / 1.850 | 0 / 1.738 | 0 / 1.738 |
| 5× | 214 / 2.154 | 42 / 2.336 | 21 / 1.932 | 21 / 1.932 |

Il massimo regolabile in gioco è 2,5×. A 5× falliscono tutti: è il limite
della velocità della racchetta, non della rete.

**Difetto corretto durante l'esperimento.** Nella prima esecuzione gli
intercetti arrivavano a ±28.000 px (45 riflessioni): un urto sulla calotta
arrotondata lasciava la palla quasi verticale (seed 28: `|vx|` = 6,5 px/s, ~150 s
simulati per attraversare il campo). Ora la direzione in uscita da una racchetta
resta entro 60° dall'orizzontale (`MAX_BOUNCE_ANGLE_DEG`), con modulo e spin
conservati. Stress 100 seed × 10.000 passi dopo la correzione:

| Servizio | Rete con previsione | Inseguimento senza previsione |
| --- | ---: | ---: |
| 1× | 0 / 2.962 | 0 / 3.023 |
| 3× | **0 / 8.961** | **58 / 9.589** |

## E2 — Reti addestrate per imitazione

[`e2_trained.py`](../experiments/e2_trained.py) ·
[risultati](experiments/e2_trained.json)

Imitazione supervisionata (behaviour cloning) della politica analitica, Adam,
etichette esatte e illimitate.

**Pong** (dominio ridotto, 3 inizializzazioni, 20.000 passi; partite 10 seed ×
5.000 passi):

| Architettura | Parametri | Errore max | Direzione sbagliata | Mancate 1× | Mancate 3× |
| --- | ---: | ---: | ---: | ---: | ---: |
| analitica 2-2-2-1 | 15 | 0 | 0% | 0 | 0 |
| addestrata 2-2-2-1 | 15 | 316–629 | 24–50% | 124–129 | 329–337 |
| addestrata 2-8-1 | 33 | 8–13 | 0% | 0 | 0 |
| addestrata 2-32-1 | 129 | 7 | 0% | 0 | 0 |
| addestrata 2-32-32-1 | 1.185 | 6–8 | 0% | 0 | 0 |

La soluzione esatta a 15 parametri esiste, ma la discesa del gradiente non la
trova (minimo locale con unità morte). Con poco margine in più basta
un'approssimazione da ~7 px, che in partita non si distingue.

**Snake** (109.903 stati da 40 partite per l'addestramento, 27.868 da 10 partite
diverse per il test, 15 epoche; 20 partite su seed nuovi). Uno stato
`(corpo, mela)` che si ripete indica un ciclo infinito esatto: politica e campo
sono deterministici fino alla mela successiva.

| Architettura | Parametri | Acc. addestramento | Acc. test | Vittorie | In loop | Mele medie |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| analitica sparsa | 7.166 (991.006 dense) | 100% | 100% | 20/20 | 0 | 248 |
| 1008-64-4 | 64.836 | 95,9% | 94,1% | 0/20 | 20 | 2,1 |
| 1008-256-4 | 259.332 | 98,3% | 94,7% | 0/20 | 20 | 0,0 |
| 1008-256-256-4 | 325.124 | 100% | 94,7% | 0/20 | 19 | 2,1 |

Il 5% di errori porta il serpente in stati mai visti, dove ripete un giro senza
mangiare: è l'accumulo d'errore tipico del behaviour cloning. E6 prova a
correggerlo.

## E6 — DAgger per Snake

[`e6_dagger.py`](../experiments/e6_dagger.py) ·
[risultati](experiments/e6_dagger.json)

DAgger (Ross, Gordon e Bagnell, 2011) corregge lo spostamento di distribuzione
del behaviour cloning: l'allievo gioca, il maestro etichetta gli stati che
l'allievo visita davvero, i dati si aggregano e l'addestramento continua. Ogni
iterazione: 20 partite nuove fino a 3.000 mosse, a ogni passo agisce il maestro
con probabilità β = 0,5^i, altrimenti l'allievo. Quando l'allievo entra in un
ciclo, il maestro guida fino alla mela successiva: senza questo accorgimento le
partite si fermavano subito e ogni iterazione aggiungeva solo 2–5 mila stati,
tutti di inizio partita. Rete 1008-256-256-4 (325.124 parametri), 3 epoche per
iterazione sui dati aggregati, valutazione sulle stesse 20 partite di E2 con il
solo allievo.

| Iterazione | Stati nel dataset | Accordo sulle proprie partite | Acc. su stati del maestro | Vittorie | Morti | In loop | Mele medie (max) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (= imitazione) | 109.903 | 85,6% | 94,1% | 0 | 0 | 20 | 2,2 (8) |
| 1 | 128.159 | 86,2% | 94,5% | 0 | 0 | 20 | 2,4 (6) |
| 2 | 144.318 | 90,1% | 94,2% | 0 | 1 | 19 | 5,0 (18) |
| 3 | 165.952 | 93,2% | 95,0% | 0 | 4 | 16 | 9,1 (35) |
| 4 | 182.253 | 94,6% | 95,4% | 0 | 9 | 11 | 16,4 (37) |
| 5 | 198.398 | 96,4% | 95,4% | 0 | 7 | 13 | 22,2 (41) |
| 6 | 212.912 | 96,4% | 95,7% | 0 | 13 | 7 | 25,4 (45) |
| 7 | 232.707 | 96,8% | 96,0% | 0 | 15 | 5 | 26,3 (44) |
| 8 | 250.014 | 96,3% | 96,4% | 0 | 16 | 4 | 29,9 (45) |
| 9 | 266.337 | 97,0% | 96,5% | 0 | 13 | 7 | 30,2 (50) |
| 10 | 280.319 | — | 96,6% | 0 | 14 | 6 | 29,1 (49) |

L'"accordo sulle proprie partite" è misurato durante la raccolta
dell'iterazione, quindi in parte con il maestro al volante.

**Lettura.** DAgger funziona come previsto: elimina buona parte dei cicli e
moltiplica per 13 le mele raccolte. Ma si stabilizza intorno a 30 mele su 248,
con il 97% di mosse giuste. Il motivo è aritmetico: una partita vinta richiede
circa 8.500 mosse; con il 3% di errori, anche se solo una piccola parte fosse
fatale, la probabilità di arrivare in fondo resta praticamente nulla. In Snake
servono politiche quasi perfette, e l'imitazione approssimata non ci arriva.

**Limiti.** Una sola architettura densa, 10 iterazioni, ~25 minuti di CPU.
Architetture convoluzionali, più dati o apprendimento per rinforzo potrebbero
fare meglio; l'esperimento dice che con un maestro perfetto e DAgger una MLP
generica non basta in questo budget, non che una rete addestrata non possa
completare Snake.

## E3 — Robustezza al rumore e alla precisione

[`e3_noise.py`](../experiments/e3_noise.py) ·
[risultati](experiments/e3_noise.json)

Ogni coefficiente memorizzato diventa `w·(1 + σ·ε)`, ε ~ N(0,1): gli zeri
strutturali restano zero e σ si legge come precisione relativa. Pong è giocato a
3× (vedi E1), Snake su 5 partite per livello.

| σ | Pong analitica: errore max / mancate | Pong addestrata 1.185: errore max / mancate | Snake scorciatoie: decisioni uguali / vittorie | Snake ciclo: decisioni uguali / vittorie |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 / 0 | 7,8 / 0 | 100% / 5 | 100% / 5 |
| 10⁻⁶ | 0,007 / 0 | 7,8 / 0 | 100% / 5 | 100% / 5 |
| 10⁻⁵ | 0,07 / 0 | 7,8 / 0 | 99,5% / 5 | 100% / 5 |
| 10⁻⁴ | 0,7 / 0 | 7,8 / 0 | 85,6% / **0** | 100% / 5 |
| 10⁻³ | 7,1 / 0 | 7,9 / 0 | 85,2% / 0 | 100% / 5 |
| 10⁻² | 71 / 0 | 13 / 0 | 48% / 0 | 70% / **0** |
| 10⁻¹ | 700 / **357** | 92 / 0 | 23% / 0 | 36% / 0 |

**Previsioni fatte prima della misura.**
- Snake con scorciatoie: le differenze di rango (fino a 251) passano per un
  gradino `ReLU(N+1−Z) − ReLU(N−Z)` moltiplicato per N = 252, quindi un errore
  δ diventa ~252·δ. Soglia stimata ≈ 5·10⁻⁶; osservata fra 10⁻⁵ e 10⁻⁴.
- Ciclo: margini dell'ordine di 1 su somme di circa 180 → soglia ≈ 3·10⁻³;
  osservata fra 10⁻³ e 10⁻².
- Pong: errore ≈ 7.000·σ px, misurato esattamente lineare.

**Precisione numerica** (pesi e uscita di ogni strato arrotondati):

| Formato | Pong: errore max / mancate 3× | Snake scorciatoie: decisioni uguali / vittorie |
| --- | ---: | ---: |
| float32 | 0,0001 / 0 | 100% / 5 |
| float16 | 1,07 / 0 | 100% / 5 |
| bfloat16 | 17 / 0 | 99,2% / **0** |

I coefficienti di Snake sono interi ≤ 253, esatti in float16. bfloat16 ha 8 bit
di mantissa e li arrotonda; in float16 i valori penalizzati vanno in overflow a
−∞, ma l'argmax resta corretto.

**Conclusioni.** I pesi analitici non sono robusti per natura: dipende dai
margini che la costruzione lascia. Snake con scorciatoie moltiplica
coefficienti grandi e non ha margine; il ciclo ne ha cento volte di più. Contro
rumore forte la rete addestrata, più "liscia", regge meglio. Lo 0,8% di
decisioni diverse basta a impedire ogni vittoria: in Snake l'errore non resta
locale (come conferma E6).

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

Il greedy non muore mai, ma in 23 partite su 30 resta a seguire la coda fino al
limite (almeno 239 mele per partita, 245,5 in media). Il rapporto con il limite
di Manhattan misura le deviazioni sulla sequenza di mele effettivamente vista,
non la distanza dall'ottimo.

**La seconda metà non migliora nulla.** Da metà campo occupato la rete torna al
ciclo (31,8 mosse/mela contro 31,4 della baseline): tutto il −46% viene dalla
prima metà.

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

## E5 — Costo della forma neurale

[`e5_timing.py`](../experiments/e5_timing.py) ·
[risultati](experiments/e5_timing.json)

Stessa decisione come rete e come codice Python. La regola Python di Snake
([`policies.py`](../experiments/policies.py)) è stata prima verificata identica
ai logit della rete su 1.546 stati.

| Decisione | Rete | Python | Rapporto |
| --- | ---: | ---: | ---: |
| Pong, singola | 8,0 µs | 0,62 µs | 13× |
| Pong, batch NumPy (per elemento) | 0,009 µs | 0,021 µs | **0,4×** |
| Snake scorciatoie, singola | 233 µs (181 di codifica e validazione) | 11,9 µs | 20× |
| Snake ciclo, singola | 7,9 µs | 0,08 µs (tabella) | 100× |

I valori assoluti variano fra esecuzioni; i rapporti sono stabili (13×/20×/100×
qui, 11×/21×/110× in un'esecuzione precedente).

## Confronto con l'articolo di partenza

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 novembre 2026; post di blog,
"very early results", non sottoposto a revisione).

| | Articolo | Questo progetto |
| --- | --- | --- |
| Chi scrive i pesi | Un LLM ("GPT-6 Astra"), direttamente come numeri o con codice in una sandbox, senza ottimizzatore né gradienti | Codice Python che traduce un algoritmo noto; il progetto non documenta se l'abbia scritto un LLM o una persona |
| Conoscenza di partenza | Regole o sorgente dell'ambiente, a volte un episodio registrato | L'algoritmo completo (ciclo hamiltoniano con scorciatoie, Tapsell 2015) |
| Riferimento addestrato | PPO, 100M e 500M passi | Imitazione supervisionata e DAgger del maestro analitico |
| Giochi | MinAtar Breakout, Seaquest, Asterix; Go 9×9; Kinetix | Pong e Snake |
| Robustezza e precisione | Non studiate | E3 |
| Costo (token, tempo) | Non riportato | Solo tempo macchina; tempo di progettazione non misurato |

Nell'articolo il risultato contro PPO è misto. Punteggi medi su 1.000 partite,
PPO 100M / PPO 500M / pesi scritti dall'LLM:
Breakout 15,8 / 25,3 / **517,4**; Seaquest 5,6 / **88,9** / 53,1; Asterix
25,8 / **55,2** / 53,2. L'LLM supera PPO a 500M passi solo in Breakout.

## Che cosa si può concludere

1. **Pong non prova la rete a velocità normale.** Il benchmark ora è a 3×,
   dove la previsione fa la differenza (0 contro 58 mancate).
2. **"Tutto nei pesi" ha un prezzo misurabile:** 5 parametri per riflessione in
   Pong; in Snake, nessun margine numerico (rumore 10⁻⁴, bfloat16).
3. **Il valore dei pesi analitici è la garanzia.** Completamento su ogni seed e
   dimensione provati, dove imitazione e DAgger, con lo stesso maestro perfetto,
   non vincono una partita.
4. **Non dimostra che un LLM batta l'addestramento.** Qui nessun LLM ha scritto
   pesi in modo verificato; l'algoritmo era noto; le reti addestrate imparano
   dal maestro analitico, quindi presuppongono la soluzione; il tempo di
   progettazione non è misurato; non è stato provato l'apprendimento per
   rinforzo. Per sostenere quella tesi servirebbero: problemi senza algoritmo
   noto, più tentativi indipendenti dell'LLM con tasso di successo, confronto con
   RL forte a parità di budget, verifica indipendente e conteggio dei costi
   (token e iterazioni contro ore di calcolo e dati).
5. **Possibile miglioramento di Snake:** scorciatoie sicure anche con il corpo
   oltre metà campo, dove oggi non si guadagna nulla.
