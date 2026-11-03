"""Real SDL rendering, input and frame-independent presentation regressions."""

import math
import os
import unittest
from unittest.mock import patch

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame

from pong import PongApp


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = PongApp(seed=42)

    def tearDown(self):
        pygame.quit()

    def key(self, key):
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    @staticmethod
    def state(app):
        game = app.game
        return (game.ball_x, game.ball_y, game.ball_vx, game.ball_vy,
                game.player_y, game.opponent_y, game.stats.player_score,
                game.stats.opponent_score, game.stats.frames)

    def test_diagnostics_start_hidden_and_open_outside_playing_field(self):
        self.assertFalse(self.app.show_graph)
        self.app.draw()
        self.assertIsNone(self.app.diagnostics_rect)
        self.key(pygame.K_g)
        self.app.draw()
        self.assertTrue(self.app.show_graph)
        self.assertFalse(self.app.court_rect.colliderect(self.app.diagnostics_rect))
        self.assertGreaterEqual(self.app.diagnostics_rect.left,
                                self.app.court_rect.right)
        self.assertEqual(self.app.canvas.get_size(), (1440, 900))
        self.key(pygame.K_g)
        self.assertEqual(self.app.canvas.get_size(), (1200, 900))

    def test_one_second_has_same_physics_at_30_60_and_144_render_frames(self):
        states = []
        for fps in (30, 60, 144):
            app = PongApp(seed=42)
            for _ in range(fps):
                app.update(1 / fps)
            self.assertAlmostEqual(app.game.elapsed, 1.0)
            self.assertEqual(app.game.stats.frames, 120)
            states.append(self.state(app))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])

    def test_pause_freezes_physics_and_render_position_then_resumes(self):
        self.app.update(0.021)
        before = self.state(self.app)
        self.app.draw()
        position = self.app.ball_draw_position
        self.key(pygame.K_SPACE)
        self.app.update(3.0)
        self.app.draw()
        self.assertTrue(self.app.paused)
        self.assertEqual(self.state(self.app), before)
        self.assertEqual(self.app.ball_draw_position, position)
        self.key(pygame.K_SPACE)
        self.app.update(1 / 60)
        self.assertFalse(self.app.paused)
        self.assertNotEqual(self.state(self.app), before)

    def test_restart_replays_original_seed_and_discards_pending_time(self):
        initial = self.state(self.app)
        self.app.update(0.5)
        progressed = self.state(self.app)
        self.app.update(0.003)
        self.key(pygame.K_r)
        self.assertEqual(self.state(self.app), initial)
        self.assertFalse(self.app.trail)
        self.app.update(0.5)
        self.assertEqual(self.state(self.app), progressed)

    def test_visible_ball_and_capsules_fit_the_physical_wall_bounds(self):
        game = self.app.game
        game.ball_x, game.ball_y = 500.0, 16.0
        game.player_y, game.opponent_y = 59.0, 591.0
        self.app.draw()
        self.assertEqual(self.app.world_to_canvas(0, 7), (100, 132))
        self.assertEqual(self.app.world_to_canvas(1000, 643), (1100, 768))
        self.assertEqual(self.app.ball_draw_position, (600, 141))
        left, right = self.app.paddle_rects
        self.assertEqual(left, pygame.Rect(134, 132, 14, 104))
        self.assertEqual(right, pygame.Rect(1052, 664, 14, 104))
        self.assertEqual(left.top, self.app.court_rect.top)
        self.assertEqual(right.bottom, self.app.court_rect.bottom)
        self.assertEqual(self.app.ball_draw_position[1] - 9,
                         self.app.court_rect.top)

    def test_interpolation_keeps_ball_between_neighboring_physics_states(self):
        game = self.app.game
        game.ball_x, game.ball_y = 500.0, 325.0
        game.ball_vx, game.ball_vy = 360.0, 0.0
        self.app.update(1 / 120 + 1 / 240)
        self.app.draw()
        self.assertAlmostEqual(game.ball_x, 503.0)
        self.assertAlmostEqual(self.app.ball_draw_position[0], 601.5)

    def test_resize_letterboxes_and_mouse_buttons_use_window_coordinates(self):
        self.app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=600, h=600))
        self.app.draw()
        self.assertEqual(self.app.viewport_rect, pygame.Rect(0, 75, 600, 450))
        self.assertIsNone(self.app.window_to_canvas((20, 20)))
        self.assertEqual(self.app.window_to_canvas((300, 300)), (600, 450))
        self.app.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(85, 489)))
        self.assertTrue(self.app.paused)
        self.app.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(85, 489)))
        self.assertFalse(self.app.paused)

    def test_startup_fits_small_desktop_and_keeps_footer_buttons_reachable(self):
        with patch('pygame.display.get_desktop_sizes', return_value=[(1366, 768)]):
            app = PongApp(seed=42)
        app.draw()
        width, height = app.screen.get_size()
        self.assertLessEqual(width, 1286)
        self.assertLessEqual(height, 668)
        self.assertEqual(app.canvas.get_size(), (1200, 900))
        for name in ('pause', 'restart', 'diagnostics'):
            center = app.button_rects[name].center
            position = (round(app.viewport_rect.x + center[0] * app.viewport_rect.width / 1200),
                        round(app.viewport_rect.y + center[1] * app.viewport_rect.height / 900))
            self.assertTrue(app.screen.get_rect().collidepoint(position))
            app.handle_event(pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=position))
            if name == 'pause':
                self.assertTrue(app.paused)
            elif name == 'restart':
                self.assertFalse(app.paused)
            else:
                self.assertTrue(app.show_graph)

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        self.assertEqual(app.screen.get_size(), (width, height))

    def test_startup_does_not_upscale_the_window_on_a_large_desktop(self):
        with patch('pygame.display.get_desktop_sizes', return_value=[(3840, 2160)]):
            app = PongApp(seed=42)
        self.assertEqual(app.screen.get_size(), (1200, 900))

    def test_speed_keys_change_ball_velocity_without_changing_elapsed_time(self):
        before = math.hypot(self.app.game.ball_vx, self.app.game.ball_vy)
        self.key(pygame.K_EQUALS)
        self.assertAlmostEqual(math.hypot(self.app.game.ball_vx, self.app.game.ball_vy),
                               before * 1.1)
        self.key(pygame.K_MINUS)
        self.assertAlmostEqual(math.hypot(self.app.game.ball_vx, self.app.game.ball_vy),
                               before * 1.1 / 1.1)
        self.app.update(0.25)
        self.assertAlmostEqual(self.app.game.elapsed, 0.25)

    def test_score_clears_trail_and_serves_without_interpolating_across_court(self):
        self.app.update(0.1)
        self.assertTrue(self.app.trail)
        game = self.app.game
        game.ball_x, game.ball_y = -8.5, 200.0
        game.ball_vx, game.ball_vy = -360.0, 0.0
        self.app.update(1 / 120)
        self.app.draw()
        self.assertEqual(game.stats.opponent_score, 1)
        self.assertGreater(game.serve_remaining, 0)
        self.assertFalse(self.app.trail)
        self.assertEqual(self.app.ball_draw_position, (600, 450))
        self.app.update(0.1)
        self.assertFalse(self.app.trail)
        self.assertEqual((game.ball_x, game.ball_y), (500, 325))

    def test_restart_and_diagnostics_buttons_work_in_scaled_sidebar_layout(self):
        self.app.update(0.5)
        self.key(pygame.K_g)
        self.app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=720, h=450))
        self.app.draw()
        self.assertEqual(self.app.viewport_rect, pygame.Rect(0, 0, 720, 450))
        self.app.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(170, 413)))
        self.assertEqual(self.app.game.stats.frames, 0)
        self.assertTrue(self.app.show_graph)
        self.app.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(505, 413)))
        self.assertFalse(self.app.show_graph)

    def test_invalid_elapsed_time_does_not_advance_the_game(self):
        before = self.state(self.app)
        for dt in (-1, float('nan'), float('inf')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.app.update(dt)
        self.assertEqual(self.state(self.app), before)

    def test_fullscreen_returns_to_the_previous_window_size(self):
        original_size = self.app.screen.get_size()
        self.key(pygame.K_f)
        self.assertTrue(self.app.fullscreen)
        self.app.draw()
        self.key(pygame.K_f)
        self.assertFalse(self.app.fullscreen)
        self.assertEqual(self.app.screen.get_size(), original_size)

    def test_escape_and_window_close_stop_the_app(self):
        self.key(pygame.K_ESCAPE)
        self.assertFalse(self.app.running)
        app = PongApp(seed=42)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        app.run()
        self.assertFalse(app.running)
        self.assertFalse(pygame.get_init())


if __name__ == '__main__':
    unittest.main()
