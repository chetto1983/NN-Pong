# Fonti degli asset Snake

Riferimento visivo acquisito il 3 novembre 2026: https://snake-game.io/,
gioco Classic, tema Sunset, serpente Blue. Questa è una demo locale indipendente.

I seguenti file sono copie locali degli asset caricati dalla pagina di riferimento:

| File | URL sorgente |
| --- | --- |
| `apple_types.webp` | https://snake-game.io/game/snake/images/apple_types.webp?v=7 |
| `trophy_types.webp` | https://snake-game.io/game/snake/images/trophy_types.webp?v=7 |
| `help_white_24dp.png` | https://snake-game.io/game/snake/images/help_white_24dp.png?v=7 |
| `settings_white_24dp.png` | https://snake-game.io/game/snake/images/settings_white_24dp.png?v=7 |
| `fullscreen_white_24dp.png` | https://snake-game.io/game/snake/images/fullscreen_white_24dp.png?v=7 |
| `volume_up_white_24dp.png` | https://snake-game.io/game/snake/images/volume_up_white_24dp.png?v=7 |
| `refresh_white_24dp.png` | https://snake-game.io/game/snake/images/refresh_white_24dp.png?v=7 |

`Nunito-Bold.ttf` è la conversione senza modifica dei glifi del font WOFF2
https://snake-game.io/themes/snake-game.io_v3/resources/fonts/Nunito-Bold.woff2?v=2.
La conversione a TTF serve alla compatibilità con SDL_ttf/Pygame.
La licenza SIL Open Font License è in `OFL-Nunito.txt`, recuperata da
https://github.com/google/fonts/blob/main/ofl/nunito/OFL.txt.

Mela e trofeo vengono selezionati dai rispettivi sprite sheet durante il rendering.
Gli asset del sito conservano i diritti dei rispettivi titolari; non viene
attribuita loro una nuova licenza da questo progetto.

Il serpente nel riferimento è disegnato proceduralmente in un canvas; la demo
usa un renderer Pygame scritto per riprodurne forma, colori, ombre e occhi.
Il codice del motore originale del sito non è incluso nella repository.
