# Snake Classic — verifica della grafica

**final result: passed**

## Evidenze

- Fonte: https://snake-game.io/, acquisita il 3 novembre 2026 con browser reale.
- Source visual truth: `docs/images/snake-reference.png`.
- Implementazione: `docs/images/snake.png`, catturata dal renderer Pygame con
  driver Windows nativo; finestra nascosta durante la verifica automatica.
- Entrambe le immagini sono **908×801 px**, densità 1:1, senza ridimensionamento.
- Screenshot della sorgente ritagliato dal browser sull'elemento del gioco:
  il portale web circostante non è parte del target.
- Stato comparato: Classic/Sunset, testa in (4,7), corpo di quattro celle,
  mela in (13,7), punteggio 0. Record impostato a 1 soltanto nella cattura
  dell'implementazione per corrispondere alla sorgente.
- Confronto completo eseguito presentando entrambe le immagini nella stessa
  chiamata di visualizzazione, non soltanto leggendo il codice.
- Dettagli controllati nella stessa vista a piena risoluzione: testa/occhi,
  contorno del corpo, mela, trofeo, quattro icone e font del punteggio.
  A questa risoluzione questi dettagli sono leggibili senza ulteriori crop.
- Stati aggiuntivi verificati: aiuto, impostazioni, pausa, movimento e finestra
  ridotta a 454×500 con letterboxing. Acquisizioni di lavoro in `output/playwright/`.
- Catturata anche la sorgente a viewport mobile 390×844 come riferimento;
  la consegna richiesta è un'app desktop Pygame, non un port per telefono.

## Risultati per superficie

| Superficie | Verifica |
| --- | --- |
| Font e tipografia | Nunito Bold originale, convertito da WOFF2 a TTF; dimensione del punteggio 22 px, leggibile e allineato alla barra. |
| Spaziatura e layout | Barra 70 px; campo 18×14, celle 48 px, origine (22,100); icone nelle posizioni misurate. Il ridimensionamento conserva proporzioni e click. |
| Colori | Cornice, barra e due colori delle celle corrispondono ai campioni della sorgente. |
| Asset | Stessi sprite locali per mela/trofeo/icone; serpente procedurale Pygame con forma, occhi e stile del riferimento. Nessuna risorsa caricata dalla rete durante il gioco. |
| Contenuti | Punteggio/record corrispondenti. Aiuto e impostazioni adattati in italiano al controllo automatico e alla sola modalità Classic. |

Nessuna differenza P0/P1/P2 rimasta nel perimetro della demo concordata.
Il pannello delle impostazioni è volutamente limitato a velocità e percorso:
non dichiara disponibili le altre modalità o personalizzazioni del portale.

## Confronti e correzioni

1. Primo confronto: geometria e palette corrispondenti. Il trofeo mancava nella
   cattura perché il record dell'app nuova era zero; normalizzato il solo stato
   della cattura a record 1, senza cambiare il comportamento dell'app.
2. Verifica funzionale: corretta la X della pausa per riprendere la partita;
   la X non viene mostrata nello stato terminale. Test di regressione passato.
3. Nuova cattura Windows a 908×801: confronto superato. Testati riavvio,
   pausa/ripresa, aiuto, click con finestra ridotta e uscita.
4. Revisione indipendente: risolti sovrapposizione tra punteggio a tre cifre
   e trofeo, e pulsante Gioca nei pannelli aperti a partita terminata.
   Entrambe le regressioni osservate prima della correzione e poi superate.
   Ricatturati stato iniziale e partita a 100 mele/104 segmenti; quest'ultima
   evidenza è in `output/playwright/snake-three-digit-score.png`.

## Interazioni e controlli

- Suite completa: 54 test superati, inclusi 17 test UI reali con SDL dummy.
- Windows nativo: 180 frame, pausa/ripresa, aiuto, riavvio, ridimensionamento,
  click dopo la trasformazione delle coordinate e uscita verificati.
- In suite: velocità limitata, percorso, mute, full screen/ripristino finestra,
  replay, record persistente e recupero da record corrotto.
- Nessun traceback o errore Pygame nelle prove. La console del browser sorgente
  riportava zero errori; la consegna non usa una console browser.

## Follow-up polish

- P3: antialiasing, piccoli riflessi del corpo e ombre degli occhi differiscono
  leggermente tra Canvas del sito e renderer Pygame. Non si dichiara una copia
  pixel per pixel.
- Le impostazioni ridotte e il messaggio di vittoria a campo pieno sono scelte
  funzionali della demo, non stati replicati dal portale originale.

## Checklist

- [x] Acquisizione sorgente e screenshot implementazione aperti e confrontati.
- [x] Palette, font, asset e misure verificati.
- [x] Comandi visibili funzionanti e testo senza tagli nei pannelli.
- [x] Rendering nativo Windows e ridimensionamento verificati.
- [x] Nessun P0/P1/P2 aperto nel perimetro Classic/Pygame.
