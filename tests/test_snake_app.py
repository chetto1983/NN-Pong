"""Exercise the real Snake window with SDL's offscreen video/audio drivers."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame

from snake_game.app import SnakeApp


class SnakeAppTests(unittest.TestCase):
    def setUp(self):
        self.app = SnakeApp(seed=0)

    def tearDown(self):
        pygame.quit()

    def key(self, key):
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def click(self, pos):
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))

    def test_render_uses_reference_board_and_visible_snake_and_apple(self):
        self.app.draw()
        screen = self.app.screen
        self.assertEqual(screen.get_size(), (908, 801))
        self.assertEqual(screen.get_at((450, 15))[:3], (100, 77, 31))
        self.assertEqual(screen.get_at((10, 400))[:3], (183, 80, 48))
        self.assertEqual(screen.get_at((46, 124))[:3], (251, 193, 78))
        self.assertEqual(screen.get_at((94, 124))[:3], (252, 178, 79))
        red, green, blue = screen.get_at((166, 460))[:3]
        self.assertGreater(blue, red + 80)
        red, green, blue = screen.get_at((670, 460))[:3]
        self.assertGreater(red, green + 50)

    def test_pause_stops_clock_and_arrow_resumes_network(self):
        self.key(pygame.K_SPACE)
        self.app.update(0.5)
        self.assertEqual(self.app.game.steps, 0)
        self.key(pygame.K_LEFT)
        self.app.update(0.11)
        self.assertEqual(self.app.game.steps, 1)
        self.assertEqual(self.app.game.body[0], (5, 7))

    def test_restart_restores_score_seed_and_selected_network(self):
        net = self.app.game.net
        self.app.update(0.95)
        self.assertGreater(self.app.game.steps, 0)
        self.key(pygame.K_r)
        self.assertEqual(self.app.game.steps, 0)
        self.assertEqual(self.app.game.score, 0)
        self.assertEqual(self.app.game.body[0], (4, 7))
        self.assertIs(self.app.game.net, net)

    def test_speed_is_bounded_and_path_toggle_works(self):
        for _ in range(150):
            self.key(pygame.K_MINUS)
        self.assertEqual(self.app.speed, 1)
        for _ in range(150):
            self.key(pygame.K_EQUALS)
        self.assertEqual(self.app.speed, 120)
        self.key(pygame.K_g)
        self.assertTrue(self.app.show_path)
        self.app.draw()

    def test_help_pauses_and_escape_restores_previous_pause_state(self):
        self.click((767, 35))
        self.assertEqual(self.app.overlay, 'help')
        self.app.update(0.5)
        self.assertEqual(self.app.game.steps, 0)
        self.key(pygame.K_ESCAPE)
        self.assertIsNone(self.app.overlay)
        self.assertTrue(self.app.running)
        self.app.update(0.11)
        self.assertEqual(self.app.game.steps, 1)
        self.key(pygame.K_SPACE)
        self.key(pygame.K_h)
        self.key(pygame.K_ESCAPE)
        self.assertTrue(self.app.paused)

    def test_settings_speed_path_and_play_buttons_work(self):
        self.click((805, 35))
        self.assertEqual(self.app.overlay, 'settings')
        self.app.draw()
        self.click((588, 363))
        self.assertEqual(self.app.speed, 20)
        self.click((454, 432))
        self.assertTrue(self.app.show_path)
        self.click((454, 548))
        self.assertIsNone(self.app.overlay)
        self.assertFalse(self.app.paused)

    def test_scaled_window_maps_header_click_correctly(self):
        self.app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=454, h=500))
        self.app.draw()
        self.click((402, 67))
        self.assertEqual(self.app.overlay, 'settings')

    def test_mute_button_changes_audio_state(self):
        enabled = self.app.sound_enabled
        self.click((881, 35))
        self.assertEqual(self.app.sound_enabled, not enabled)
        self.app.draw()

    def test_fullscreen_button_and_key_return_to_window(self):
        self.click((843, 35))
        self.assertTrue(self.app.fullscreen)
        self.app.draw()
        self.key(pygame.K_f)
        self.assertFalse(self.app.fullscreen)
        self.assertEqual(self.app.screen.get_size(), (908, 801))

    def test_terminal_replay_and_pause_play_buttons_resume_a_game(self):
        self.app.game.step(3)  # Reversing into the body ends the game.
        self.assertEqual(self.app.game.status, 'dead')
        self.app.draw()
        self.click((454, 548))
        self.assertEqual(self.app.game.status, 'running')
        self.assertEqual(self.app.game.steps, 0)
        self.key(pygame.K_SPACE)
        self.app.draw()
        self.click((454, 548))
        self.assertFalse(self.app.paused)

    def test_close_button_on_pause_card_resumes(self):
        self.key(pygame.K_SPACE)
        self.click((634, 240))
        self.assertFalse(self.app.paused)

    def test_three_digit_score_is_not_overpainted_by_record(self):
        self.app.game.score = 248
        self.app.best_score = 248
        self.app.draw()
        font = pygame.font.Font(str(Path(__file__).resolve().parents[1]
                                    / 'snake_game' / 'assets' / 'Nunito-Bold.ttf'), 22)
        text = font.render('248', True, (255, 255, 250))
        expected = pygame.Surface(text.get_size())
        expected.fill((100, 77, 31))
        expected.blit(text, (0, 0))
        actual = self.app.canvas.subsurface((52, 18, *text.get_size()))
        self.assertEqual(pygame.image.tobytes(actual, 'RGB'),
                         pygame.image.tobytes(expected, 'RGB'))

    def test_play_from_terminal_help_or_settings_starts_a_new_game(self):
        for status in ('dead', 'won'):
            for position in ((767, 35), (805, 35)):
                with self.subTest(status=status, position=position):
                    self.app.game.status = status
                    self.app.game.score = 248 if status == 'won' else 0
                    self.click(position)
                    self.click((454, 548))
                    self.assertEqual(self.app.game.status, 'running')
                    self.assertEqual(self.app.game.score, 0)
                    self.assertEqual(self.app.game.steps, 0)
                    self.assertFalse(self.app.paused)
                    self.assertIsNone(self.app.overlay)

    def test_corrupt_record_and_disabled_audio_do_not_prevent_play(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text('{invalid JSON', encoding='utf-8')
            app = SnakeApp(seed=0, record_path=path)
            self.assertEqual(app.best_score, 0)
            pygame.mixer.quit()
            app.update(0.95)
            self.assertEqual(app.game.score, 1)
            app.draw()

    def test_nonfinite_record_falls_back_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text('{"best_score": 1e309}', encoding='utf-8')
            app = SnakeApp(seed=0, record_path=path)
            self.assertEqual(app.best_score, 0)

    def test_best_score_survives_restart_with_record_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            app = SnakeApp(seed=0, record_path=path)
            app.update(0.95)
            self.assertEqual(app.game.score, 1)
            app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
            self.assertEqual(app.best_score, 1)
            loaded = SnakeApp(seed=0, record_path=path)
            self.assertEqual(loaded.best_score, 1)

    def test_quit_event_exits_run_without_hanging(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.app.run()
        self.assertFalse(self.app.running)
        self.assertFalse(pygame.get_init())


if __name__ == '__main__':
    unittest.main()
