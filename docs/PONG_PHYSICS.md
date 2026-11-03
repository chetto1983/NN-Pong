# Modello fisico di Pong

## Geometria e tempo

Il mondo usa pixel e secondi: campo largo 1000, pareti orizzontali a y=7 e
y=643, palla di raggio 9. Il centro della palla resta quindi fra y=16 e y=634.
Le racchette sono capsule verticali larghe 14 e alte 104; il raggio delle
estremità è 7. Il renderer trasforma queste coordinate senza cambiare le
proporzioni, così il contatto visibile coincide con quello fisico.

Il rendering è separato dalla simulazione: l'app accumula il tempo reale ed
esegue passi fisici fissi da 1/120 di secondo. Lo stress test conserva il
comando storico `--frames`, dove ogni passo richiesto dura 1/60 di secondo.
La velocità iniziale è 360 px/s e quella massima delle racchette è 420 px/s.

## Collisioni continue

Per ogni tratto temporale il motore cerca il primo contatto fra palla,
pareti e racchette. Le racchette vengono espanse del raggio della palla e
la traiettoria del centro viene intersecata con i fianchi e le calotte
circolari. Il movimento verticale della racchetta entra nella velocità relativa.

La palla avanza fino all'urto, la velocità viene aggiornata e il tempo rimasto
viene simulato. In questo modo un singolo passo può contenere più rimbalzi.
Un punto già passato dietro una racchetta non viene riportato davanti al campo.

Passo fisso e tempo di impatto sono principi usati anche nei motori fisici;
si vedano la [simulazione](https://box2d.org/documentation/md_simulation.html)
e il [tempo di impatto](https://box2d.org/documentation/md_collision.html)
nella documentazione Box2D. Il progetto implementa direttamente il proprio
solver per queste geometrie e non aggiunge Box2D come dipendenza.

## Risposta dell'urto

Le pareti sono elastiche e senza attrito: si inverte la componente verticale
della velocità, mantenendo quella orizzontale e il modulo della velocità.

Sulle racchette il modello usa massa della palla pari a 1, momento d'inerzia
del disco `I = r²/2`, restituzione normale 1 e attrito tangenziale di Coulomb.
Detti `n` la normale, `t=(-n_y,n_x)` la tangente e `v_rel` la velocità della
palla rispetto alla racchetta:

```text
v_n = dot(v_rel, n)
v_t = dot(v_rel, t) - spin * r
J_n = -2 * v_n                         (solo contatti in avvicinamento)
J_t = clamp(-v_t / 3, -mu*J_n, mu*J_n)
v_nuova = v + J_n*n + J_t*t
spin_nuovo = spin - 2*J_t/r
```

Un colpo centrale contro una racchetta ferma non produce la vecchia
accelerazione automatica del 2,5%. Una racchetta in movimento può invece
trasferire energia alla palla; l'attrito può dissiparne una parte. Lo spin
influenza i contatti successivi, senza introdurre una forza curva durante il
volo. I tasti +/- sono un intervento esplicito dell'utente sulla velocità.

Per evitare che due racchette passive finiscano in scambi orizzontali, l'avversario
esegue una battuta: al punto previsto raggiungibile aggiunge un'oscillazione
verticale di ampiezza 30 px e frequenza 0,6 Hz. Il suo motore resta limitato a
357 px/s. La variazione di traiettoria deriva dalla velocità reale della
racchetta nel contatto e dalle formule sopra, non da una modifica arbitraria
della velocità della pallina. I limiti del campo possono fermare temporaneamente
la battuta; lo scambio può restare piatto per parte del ciclo.

Dopo un punto, il servizio resta al centro per 0,65 secondi di tempo simulato.
Anche questa pausa consuma il tempo residuo del passo: non viene scartato tempo
in corrispondenza del goal.

## Previsione della rete

La rete resta una ReLU 2→8→2→1 con 45 parametri analitici. Python calcola
l'intercetto grezzo sul piano frontale della racchetta, poi normalizza la
coordinata con il periodo `2*SPAN`. La rete calcola il fold e la correzione
della racchetta. La normalizzazione conserva i rimbalzi invece di tagliare
la previsione a un limite arbitrario.

Il punto previsto assume volo libero e rimbalzi sulle pareti fino al piano
frontale: un contatto anticipato con la calotta della racchetta segue invece
la normale della superficie realmente colpita.

## Limiti del modello

- È Pong in un piano senza gravità, resistenza dell'aria o deformazione della palla.
- Le racchette sono corpi cinematici: il loro motore mantiene la velocità di
  comando e non subisce rinculo. Nel caso limite di una palla compressa fra
  racchetta e sponda, il motore cede e si arresta: la componente verticale
  bloccata della palla viene annullata, senza spostarla attraverso le superfici.
  Un avvicinamento laterale viene riflesso verso l'esterno, conservando il suo
  modulo e contando una sola respinta, così la palla può uscire dal contatto.
  Questo evita una sequenza infinita di impulsi elastici allo stesso istante.
- L'interpolazione grafica approssima il movimento nell'intervallo fra due
  stati fisici consecutivi; la logica degli urti usa sempre gli stati simulati.
- Le statistiche misurano il controllore su questo modello, non una garanzia
  di vittoria contro qualsiasi avversario.
