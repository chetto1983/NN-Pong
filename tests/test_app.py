"""Rendering reale Pygame con driver SDL senza finestra."""

import os
import unittest

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pong


class AppTests(unittest.TestCase):
    def test_render_and_exit(self):
        app = pong.PongApp(seed=42)
        try:
            for _ in range(10):
                app.game.step()
                app.draw()
            self.assertEqual(app.screen.get_size(), (1000, 650))
            self.assertEqual(app.game.stats.frames, 10)
            self.assertNotEqual(app.screen.get_at((500, 300))[:3], pong.BG)
            pong.pygame.event.post(pong.pygame.event.Event(
                pong.pygame.KEYDOWN, key=pong.pygame.K_ESCAPE,
            ))
            app.run()
            self.assertFalse(app.running)
            self.assertFalse(pong.pygame.get_init())
        finally:
            pong.pygame.quit()


if __name__ == '__main__':
    unittest.main()
