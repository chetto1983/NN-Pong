# Snake: rete completa con scorciatoie

## Obiettivo autorizzato

Costruire la strategia più efficiente proposta all'utente: meno mosse per
completare il campo, pesi scritti analiticamente, nessun training. La rete
precedente da 150 parametri rimane disponibile come baseline. Pong non cambia.

## Osservazione e rete

La nuova rete riceve quattro mappe binarie della griglia, in ordine testa,
mela, coda, corpo completo. Sul campo 18×14 sono 1008 ingressi. Il packing
dello stato è soltanto una codifica: non calcola azioni ammesse, percorsi o
scorciatoie. L'uscita è costituita da quattro logits UP/RIGHT/DOWN/LEFT.

Le matrici sono sparse, memorizzate come indici e coefficienti. L'inferenza
esegue somme pesate, bias e ReLU; nessuna ricerca o sostituzione dell'azione
all'esterno della rete. La simulazione usa l'argmax anche per pesi modificati.
Il conteggio dichiarerà separatamente coefficienti memorizzati e dimensione
equivalente delle matrici dense, evitando confronti ambigui con la baseline.

## Regola da compilare nei pesi

Si conserva il ciclo hamiltoniano della baseline. Per ogni vicino della testa,
la rete calcola la distanza positiva lungo il ciclo `d`, le distanze dalla
mela `f` e dalla coda `t`, l'indicatore di bordo e l'occupazione del vicino.
La coda attuale è esclusa dall'occupazione: se ci si entra viene liberata nello
stesso passo, e non può contenere la mela.

Una mossa è ammissibile se è nel campo, non entra nel corpo, e `d <= f` e
`d <= t`. Fra queste, si sceglie la distanza maggiore: il maggior progresso
verso la mela senza superarla o superare la coda. La regola è codificata nei
logits con penalità superiori a ogni possibile distanza valida, non in un filtro
Python successivo.

Quando il corpo occupa almeno metà del campo, i pesi consentono soltanto il
successore del ciclo (`d=1`). Questo ricompatta il corpo ed evita che scorciatoie
tardive lascino buchi costosi da raggiungere. La lunghezza viene calcolata
sommando la mappa del corpo nel primo strato; anche la soglia è nella rete.

L'ordine ciclico coda→corpo→testa è conservato a partire dal corpo iniziale.
Il successore del ciclo resta una mossa disponibile; la distanza dalla mela
diminuisce finché viene raggiunta. La proprietà dipende da una mela alla volta,
crescita immediata di un segmento, campo senza ostacoli e stato iniziale coerente.

## Strati previsti

1. Gate ReLU di occupazione per ogni arco adiacente della griglia;
   sei differenze di rango traslate di N (quattro vicini, mela, coda), bordi e lunghezza.
2. Copie e soglie per riconoscere il cambio di periodo; somme delle occupazioni;
   soglie ReLU per l'attivazione delle scorciatoie prima di N/2 segmenti.
3. Distanze positive modulo N, indicatori di occupazione/bordo e gate scorciatoie.
4. Violazioni dei limiti mela/coda/densità e copie delle distanze/occupazioni.
5. Logits: `d - (N+1)*(violazione_mela + violazione_coda + violazione_densità + occupazione + bordo)`.

Prima della compilazione, un oracolo separato è stato usato per verificare la
regola: scorciatoie senza limite di densità miglioravano soltanto del 4,9% sui
seed 0–99. La soglia di metà campo è stata verificata su dieci seed di sviluppo.
Questa è progettazione e verifica dell'algoritmo; non viene addestrato un modello.

## Integrazione e verifica

- Nuova classe `ShortcutSnakeNet`, baseline `SnakeNet` invariata nei pesi.
- Protocollo `forward_state(body, food)` comune; `forward_observation(vector)`
  per esaminare direttamente l'inferenza della nuova rete.
- Simulazione e app usano di default la nuova rete; CLI consente `--policy cycle`.
- JSON sparse versione 2, caricamento versione 1 mantenuto; nuovo file pesi
  `weights/snake_shortcuts.json`.
- G visualizza il ciclo di riferimento, non una falsa previsione della politica.
- Benchmark sui seed 0–99 e separatamente 100–199, confronto con la baseline
  sulle stesse regole e seed. Le posizioni delle mele possono divergere perché
  dipendono dalle celle libere; nessuna promessa di sequenze identiche.
- Test delle gate, input incompleti, wrap del ciclo, coda mobile, crescita,
  stati densi, perturbazione dei pesi, roundtrip JSON e assenza di fallback.

Riferimento concettuale per il ciclo con scorciatoie: articolo originale di
John Tapsell, https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/.
La costruzione neurale e il codice della demo sono sviluppati per questo progetto.
