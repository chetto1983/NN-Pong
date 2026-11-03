"""Behavioral regressions for continuous, time-based Pong physics."""

import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from pong_game import PongSimulation, BALL_SPEED, PADDLE_SPEED, Stats


class ContinuousPhysicsTests(unittest.TestCase):
    def game(self, x=500, y=325, vx=0, vy=0):
        game = PongSimulation(seed=12)
        game.ball_x, game.ball_y = float(x), float(y)
        game.ball_vx, game.ball_vy = float(vx), float(vy)
        return game

    def test_free_flight_uses_pixels_per_second(self):
        game = self.game(vx=-360, vy=120)
        game.step(0.25, control=False)
        self.assertAlmostEqual(game.ball_x, 410)
        self.assertAlmostEqual(game.ball_y, 355)
        self.assertAlmostEqual(game.elapsed, 0.25)
        self.assertEqual(game.stats.frames, 1)

    def test_step_size_does_not_change_uncontrolled_motion(self):
        whole = self.game(x=200, y=320, vx=-600, vy=120)
        split = self.game(x=200, y=320, vx=-600, vy=120)
        whole.player_vy = split.player_vy = 100
        whole.step(0.8, control=False)
        for _ in range(80):
            split.step(0.01, control=False)
        for name in ('ball_x', 'ball_y', 'ball_vx', 'ball_vy',
                     'ball_spin', 'player_y', 'elapsed'):
            self.assertAlmostEqual(getattr(whole, name), getattr(split, name),
                                   places=8, msg=name)
        self.assertEqual(whole.stats.hits, split.stats.hits)

    def test_ball_behind_left_paddle_is_not_teleported(self):
        game = self.game(x=40, vx=-360)
        game.step(control=False)
        self.assertAlmostEqual(game.ball_x, 34)
        self.assertEqual(game.ball_vx, -360)
        self.assertEqual(game.stats.hits, 0)

    def test_ball_behind_right_paddle_is_not_teleported(self):
        game = self.game(x=960, vx=360)
        game.step(control=False)
        self.assertAlmostEqual(game.ball_x, 966)
        self.assertEqual(game.ball_vx, 360)

    def test_fast_ball_hits_paddle_then_consumes_remaining_time(self):
        game = self.game(x=200, vx=-12000)
        game.step(control=False)
        self.assertAlmostEqual(game.ball_x, 114)
        self.assertAlmostEqual(game.ball_vx, 12000)
        self.assertEqual(game.stats.hits, 1)
        self.assertEqual(game.stats.misses, 0)

    def test_centered_stationary_paddle_hit_preserves_speed(self):
        game = self.game(x=60, vx=-360)
        game.step(control=False)
        self.assertAlmostEqual(game.ball_x, 60)
        self.assertAlmostEqual(math.hypot(game.ball_vx, game.ball_vy), 360)
        self.assertAlmostEqual(game.ball_spin, 0)

    def test_rounded_end_deflects_along_contact_normal(self):
        game = self.game(x=70, y=270, vx=-360)
        game.step(0.08, control=False)
        self.assertEqual(game.stats.hits, 1)
        self.assertGreater(game.ball_vx, 0)
        self.assertLess(game.ball_vy, -200)
        self.assertLess(game.ball_y, 270)
        # Friction may convert translation into rotation, never add energy.
        energy_twice = (game.ball_vx**2 + game.ball_vy**2
                        + 0.5 * 9**2 * game.ball_spin**2)
        self.assertLessEqual(energy_twice, 360**2 + 1e-7)

    def test_ball_outside_rounded_end_misses(self):
        game = self.game(x=70, y=263, vx=-360)
        game.step(0.08, control=False)
        self.assertEqual(game.stats.hits, 0)
        self.assertAlmostEqual(game.ball_x, 41.2)
        self.assertEqual(game.ball_vx, -360)

    def test_moving_paddle_transfers_vertical_motion_and_spin(self):
        game = self.game(x=60, vx=-360)
        game.player_vy = 240
        game.step(control=False)
        self.assertAlmostEqual(game.ball_vx, 360)
        self.assertAlmostEqual(game.ball_vy, 80)
        self.assertAlmostEqual(game.ball_spin, -160 / 9)
        self.assertAlmostEqual(game.player_y, 329)

    def test_separating_contact_does_not_repeat_hit(self):
        game = self.game(x=57, vx=-360)
        game.step(0.01, control=False)
        self.assertEqual(game.stats.hits, 1)
        game.step(0.01, control=False)
        self.assertEqual(game.stats.hits, 1)
        self.assertAlmostEqual(game.ball_x, 64.2)

    def test_wall_reflection_preserves_speed_and_spin(self):
        game = self.game(y=17, vx=120, vy=-180)
        game.ball_spin = 15
        game.step(control=False)
        self.assertAlmostEqual(game.ball_y, 18)
        self.assertAlmostEqual(game.ball_vx, 120)
        self.assertAlmostEqual(game.ball_vy, 180)
        self.assertEqual(game.ball_spin, 15)

    def test_many_wall_bounces_in_one_step_remain_inside(self):
        game = self.game(y=325, vy=120000)
        game.step(control=False)
        # 325 + 2000, folded into [16,634], gives 179 travelling upward.
        self.assertAlmostEqual(game.ball_y, 179)
        self.assertAlmostEqual(game.ball_vy, -120000)
        self.assertGreaterEqual(game.ball_y, 16)
        self.assertLessEqual(game.ball_y, 634)

    def test_paddle_stops_at_physical_field_boundary(self):
        game = self.game()
        game.player_y, game.player_vy = 60, -PADDLE_SPEED
        game.opponent_y, game.opponent_vy = 590, PADDLE_SPEED
        game.step(0.2, control=False)
        self.assertEqual(game.player_y, 59)
        self.assertEqual(game.opponent_y, 591)
        self.assertEqual(game.player_vy, 0)
        self.assertEqual(game.opponent_vy, 0)

    def test_collision_after_paddle_stops_uses_stationary_velocity(self):
        game = self.game(x=70, y=80, vx=-360)
        game.player_y, game.player_vy = 60, -420
        game.step(0.06, control=False)
        self.assertEqual(game.stats.hits, 1)
        self.assertAlmostEqual(game.ball_vx, 360)
        self.assertAlmostEqual(game.ball_vy, 0)

    def test_prediction_normalizes_many_bounces_without_clipping(self):
        game = self.game(x=500, y=100, vx=-100, vy=10000)
        prediction = game.predict_player_intercept()
        self.assertAlmostEqual(game.raw_prediction, 44400)
        self.assertGreaterEqual(prediction, 16)
        self.assertLess(prediction, 1252)
        _, info = game.net.forward(325, prediction)
        self.assertAlmostEqual(info['fold'], 128)
        game.step(0.001)
        self.assertLess(game.output, 0)
        self.assertLess(game.player_vy, 0)

    def test_goal_scores_once_and_holds_new_serve(self):
        game = self.game(x=-8, y=100, vx=-360)
        game.step(control=False)
        self.assertEqual(game.stats.opponent_score, 1)
        self.assertEqual(game.stats.misses, 1)
        self.assertEqual((game.ball_x, game.ball_y), (500, 325))
        self.assertGreater(game.serve_remaining, 0)
        pause = game.serve_remaining
        vx, vy = game.ball_vx, game.ball_vy
        game.step(pause + 0.01, control=False)
        self.assertAlmostEqual(game.ball_x, 500 + vx * 0.01)
        self.assertAlmostEqual(game.ball_y, 325 + vy * 0.01)
        self.assertEqual(game.stats.opponent_score, 1)

    def test_serve_sets_speed_norm_and_reset_clears_game_state(self):
        game = self.game()
        game.stats.hits = 4
        game.elapsed = 12
        game.player_vy = 100
        game.ball_spin = 5
        game.serve(1)
        self.assertAlmostEqual(math.hypot(game.ball_vx, game.ball_vy), BALL_SPEED)
        self.assertGreater(game.ball_vx, 0)
        self.assertEqual(game.ball_spin, 0)
        game.reset()
        self.assertEqual(game.stats, Stats())
        self.assertEqual(game.elapsed, 0)
        self.assertEqual(game.player_vy, 0)
        self.assertEqual(game.opponent_vy, 0)
        self.assertEqual(game.serve_remaining, 0)

    def test_speed_control_limits_total_speed_preserving_direction(self):
        game = self.game(vx=-300, vy=400)
        game.scale_ball_speed(100)
        self.assertAlmostEqual(math.hypot(game.ball_vx, game.ball_vy), 900)
        self.assertAlmostEqual(game.ball_vy / game.ball_vx, -4 / 3)
        game.scale_ball_speed(0.0001)
        self.assertAlmostEqual(math.hypot(game.ball_vx, game.ball_vy), 180)
        for factor in (0, -1, math.inf, math.nan):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                game.scale_ball_speed(factor)

    def test_invalid_time_is_rejected_and_zero_time_is_noop(self):
        game = self.game(vx=360)
        game.step(0)
        self.assertEqual(game.ball_x, 500)
        self.assertEqual(game.stats.frames, 0)
        for dt in (-1, math.inf, math.nan):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                game.step(dt)

    def test_opponent_strokes_keep_late_rallies_oblique(self):
        game = PongSimulation(seed=17)
        late_vertical_speeds = []
        late_heights = []
        # Observe sustained play after two minutes; individual strokes can
        # still be clipped by a field boundary during part of their cycle.
        for frame in range(14400):
            game.step()
            if frame >= 7200:
                late_vertical_speeds.append(abs(game.ball_vy))
                late_heights.append(game.ball_y)
        self.assertGreater(max(late_vertical_speeds), 20)
        self.assertGreater(max(late_heights) - min(late_heights), 100)

    def test_opponent_can_stroke_away_from_a_wall_intercept(self):
        game = self.game(x=900, y=628, vx=360)
        game.opponent_y = 591
        game.elapsed = 1.2  # The deterministic stroke currently moves upward.
        game.step(0.01)
        self.assertLess(game.opponent_y, 591)
        self.assertLess(game.opponent_vy, 0)

    def test_control_substeps_use_the_same_physical_clock(self):
        whole = self.game(x=800, y=325, vx=360, vy=0)
        split = self.game(x=800, y=325, vx=360, vy=0)
        whole.step(0.8)
        for _ in range(96):
            split.step(1 / 120)
        for name in ('ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'ball_spin',
                     'player_y', 'opponent_y', 'opponent_vy', 'elapsed'):
            self.assertAlmostEqual(getattr(whole, name), getattr(split, name),
                                   places=8, msg=name)

    def test_wall_cap_contact_keeps_a_lateral_ball_moving(self):
        for side in ('player', 'opponent'):
            for wall_y, paddle_y in ((16, 63), (634, 587)):
                with self.subTest(side=side, wall=wall_y):
                    vx = -360 if side == 'player' else 360
                    x = 100 if side == 'player' else 900
                    game = self.game(x=x, y=wall_y, vx=vx)
                    setattr(game, side + '_y', paddle_y)
                    game.step(0.2, control=False)
                    self.assertAlmostEqual(game.ball_vx, -vx)
                    self.assertAlmostEqual(game.ball_vy, 0)
                    self.assertAlmostEqual(game.ball_y, wall_y)
                    self.assertAlmostEqual(game.elapsed, 0.2)
                    self.assertEqual(game.stats.hits, 1 if side == 'player' else 0)

    def test_wall_paddle_squeeze_stalls_actuator_without_hanging(self):
        # Isolate the regression: a zero-time collision loop must fail this
        # test by timeout rather than hang the entire test runner.
        program = '''
import json
from pong_game import PongSimulation
results = []
for paddle, center_x in (("player", 41), ("opponent", 959)):
    other = "opponent" if paddle == "player" else "player"
    for wall, center_y, velocity in ((16, 78, -420), (634, 572, 420)):
        for offset in (0, 1e-6, -1e-6):
            game = PongSimulation()
            setattr(game, paddle + "_y", center_y)
            setattr(game, paddle + "_vy", velocity)
            setattr(game, other + "_vy", 100)
            game.ball_x, game.ball_y = center_x + offset, wall
            game.ball_vx = game.ball_vy = 0
            game.ball_spin = 5
            game.step(1/60, control=False)
            hits = game.stats.hits
            game.step(0.25, control=False)
            results.append(dict(paddle=paddle, wall=wall, x=center_x+offset,
                ball_x=game.ball_x, ball_y=game.ball_y,
                vx=game.ball_vx, vy=game.ball_vy, spin=game.ball_spin,
                paddle_y=getattr(game, paddle + "_y"),
                paddle_vy=getattr(game, paddle + "_vy"),
                other_y=getattr(game, other + "_y"), elapsed=game.elapsed,
                hits=hits, final_hits=game.stats.hits))
print(json.dumps(results))
'''
        try:
            result = subprocess.run(
                [sys.executable, '-c', program], capture_output=True,
                text=True, timeout=5, cwd=Path(__file__).resolve().parents[1],
            )
        except subprocess.TimeoutExpired:
            self.fail('Wall/paddle squeeze did not terminate within 5 seconds')
        self.assertEqual(result.returncode, 0, result.stderr)
        states = json.loads(result.stdout)
        self.assertEqual(len(states), 12)
        for state in states:
            with self.subTest(paddle=state['paddle'], wall=state['wall'], x=state['x']):
                self.assertAlmostEqual(state['paddle_y'], 77 if state['wall'] == 16 else 573)
                self.assertEqual(state['paddle_vy'], 0)
                self.assertAlmostEqual(state['ball_x'], state['x'])
                self.assertAlmostEqual(state['ball_y'], state['wall'])
                self.assertEqual(state['vx'], 0)
                self.assertEqual(state['vy'], 0)
                self.assertEqual(state['spin'], 5)
                self.assertAlmostEqual(state['elapsed'], 1 / 60 + 0.25)
                self.assertAlmostEqual(state['other_y'], 325 + 100 * (1 / 60 + 0.25))
                self.assertLessEqual(state['hits'], 1)
                self.assertEqual(state['final_hits'], state['hits'])


if __name__ == '__main__':
    unittest.main()
