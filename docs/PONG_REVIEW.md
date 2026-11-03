# Verifica Pong — fisica e presentazione

## Risultato

92 test della repository superati su Windows/Python 3.13, inclusi 24 test
fisici dedicati e 15 prove dell'interfaccia Pong. La suite Snake rimane verde.

La verifica numerica dei 45 parametri della rete su 50.000 campioni mantiene
un errore massimo di `2.274e-13`.

## Grafica

- [Schermata normale](images/pong.png), canvas 1200×900.
- [Diagnostica aperta](images/pong-diagnostics.png), canvas 1440×900.
- Screenshot originali di confronto in `output/playwright/pong-before.png`.

Le immagini sono state aperte e confrontate. Campo e pareti coincidono con
le coordinate fisiche; punteggio e controlli sono separati dal campo, e la
diagnostica occupa un'area dedicata. Font leggibile, colori turchese/corallo,
racchette a capsula e scia breve sostituiscono l'interfaccia precedente.

Prova Windows nativa su 240 frame: tempo fisico, pausa/ripresa, diagnostica,
ridimensionamento, click, reset e uscita verificati. Test aggiuntivi su un
desktop simulato 1366×768 controllano finestra iniziale, footer e ripristino
da schermo intero. I risultati fisici coincidono a 30, 60 e 144 frame grafici/s.

## Fisica e revisione

Corrette collisioni dietro la racchetta, attraversamenti ad alta velocità,
rimbalzi multipli, estremità arrotondate, movimento della racchetta, punteggio
e pausa del servizio. Rimossi incremento automatico di velocità e variazione
arbitraria della componente verticale a ogni colpo.

Una revisione indipendente ha esaminato codice, test, screenshot e 240 casi
di contatto simultaneo. Il coordinatore ha poi trovato e corretto il caso di
compressione perfettamente simmetrica fra racchetta e sponda: la regressione
usa un subprocess con timeout per impedire che un futuro errore blocchi la suite.
Sono coperti anche piccoli disallineamenti e urti laterali nello stesso vincolo.

Il modello resta bidimensionale: non comprende gravità, aerodinamica o
deformazione della palla. L'arresto del motore nel caso di compressione è una
regola esplicita per una racchetta cinematica; dettagli in [PONG_PHYSICS.md](PONG_PHYSICS.md).
La matrice CI remota non è stata eseguita in questa sessione.
