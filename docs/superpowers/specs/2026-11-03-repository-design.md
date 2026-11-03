# NN Pong: repository eseguibile

## Obiettivo

Trasformare il generatore `pongNN.py` in una repository locale completa ed
eseguibile, poi eliminare l'originale dalla cartella, come richiesto dall'utente.
Il progetto conserva la rete ReLU a pesi analitici e il riferimento al weight-space
programming indicato nella conversazione.

## Approccio

Scelta implementativa: un singolo `pong.py`, coerente con il gioco esistente di
circa 460 righe, con test separati. Un package suddiviso in molti moduli aggiungerebbe
struttura senza una necessità attuale; conservare il generatore manterrebbe invece
il problema di avvio. Le decisioni di organizzazione sono dell'implementatore.

## Vincoli e risultato

- Python 3.12 o superiore; NumPy 2.5.4 e Pygame 2.6.1, già provati localmente.
- Conservare grafica, controlli e modalità `--test` / `--headless`.
- Allineare il fold della rete al centro della palla: limiti verticali 16 e 634.
- Rifiutare conteggi di frame o seed non positivi dalla CLI.
- README in italiano, dipendenze riproducibili, `.gitignore` e CI Windows/Linux.
- Test con `unittest`, senza dipendenze di sviluppo aggiuntive.
- Conservare l'originale nella cronologia Git prima della cancellazione.
- Lavorare nella cartella richiesta, che contiene una repository vuota; nessuna
  pubblicazione remota è inclusa in questa consegna.

La previsione cinematica resta esterna alla rete; non vengono aggiunti training,
nuove strategie, nuovi avversari o una riprogettazione della grafica.
