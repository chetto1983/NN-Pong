# Pong: fisica e grafica corrette

## Obiettivo

Correggere la fisica della pallina e migliorare la grafica, come richiesto
dall'utente, conservando Python/Pygame e il controllo tramite pesi analitici.

## Problemi riprodotti

- Pallina già dietro la racchetta sinistra: x=40 viene riportata a x=57 e
  conteggiata come respinta, anche senza attraversare una superficie.
- Urto centrale: la velocità passa da 6 a 6,15 senza una causa fisica.
- Grande spostamento verticale: un solo controllo di rimbalzo lascia la palla
  fuori dal campo (y=-1057).
- Le pareti della simulazione non coincidono con il rettangolo disegnato;
  grafico e HUD nascondono il gioco.

## Modello scelto

Pong bidimensionale senza gravità: pallina circolare, racchette a capsula mosse
verticalmente, pareti orizzontali elastiche. Velocità in pixel al secondo e
tempo in secondi; passi fisici fissi a 120 Hz nell'app. Ricerca continua del
primo contatto lungo lo spostamento, anche contro le estremità arrotondate
delle racchette, poi consumo del tempo rimanente. Nessun aumento automatico
della velocità a ogni respinta. Risposta sulla normale d'urto e attrito
tangenziale con spin del disco per trasferire parte del movimento della racchetta.
I tasti +/- modificano esplicitamente la velocità, entro limiti dichiarati.

Le coordinate del mondo restano 1000×650, pareti TOP=7 e BOTTOM=643, raggio=9.
La rete mantiene i suoi 45 parametri; la previsione grezza può essere ridotta
modulo il periodo di riflessione prima della rete, senza il precedente clamp
che alterava le traiettorie con molti rimbalzi.

## Grafica

Finestra ridimensionabile, layout logico 1200×900, fondo blu notte, campo
libero e delimitato dalle vere pareti, racchetta sinistra turchese e destra
corallo, pallina chiara con scia breve e discreta. Punteggio grande nell'intestazione,
comandi nel footer. Font leggibile, contorni antialias, movimento fluido.
La diagnostica è disattivata all'avvio e si apre con G in un'area separata dal
campo; nessuna linea di previsione permanente attraverso la partita.

## Struttura e compatibilità

`pong.py` resta il punto di avvio e riesporta le classi esistenti. Nuovo package
`pong_game` per separare configurazione, rete, simulazione e app. Modalità
`--test` e `--headless --seeds --frames` conservate. Snake resta disponibile.

## Verifiche

Test contro teletrasporti, collisioni mancate ad alta velocità, estremità delle
racchette, contatti ripetuti, conservazione della velocità sulle pareti e urti
centrali, influenza del movimento della racchetta, punteggio, tempi diversi
con lo stesso moto libero, previsione e riproducibilità. Catture prima/dopo
e controllo dei comandi, del ridimensionamento e della diagnostica.
