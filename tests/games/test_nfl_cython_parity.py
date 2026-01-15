import unittest

from rlcard.games.nfl.game import NFLGame
from rlcard.games.nfl.cython import CYTHON_AVAILABLE


@unittest.skipUnless(CYTHON_AVAILABLE, "Cython NFL game is not available")
class TestNFLCythonParity(unittest.TestCase):
    def _make_outcome_model(self, yards, turnover):
        class DeterministicOutcomeModel:
            def __init__(self, result):
                self._result = result
                self._used = False

            def sample(self, formation, play_type, box_count, yardline, down, ydstogo):
                if self._used:
                    raise AssertionError("Outcome model used more than once")
                self._used = True
                return self._result

        return DeterministicOutcomeModel({
            "yards_gained": int(yards),
            "turnover": bool(turnover),
        })

    def _make_special_teams(self, fg_prob=1.0, punt_yardline=60):
        class DeterministicSpecialTeams:
            def predict_fg_prob(self, yardline):
                return fg_prob

            def predict_punt_outcome(self, yardline):
                return punt_yardline

        return DeterministicSpecialTeams()

    def _make_games(self, outcome_model=None, special_teams=None):
        from rlcard.games.nfl.cython.game_fast import NFLGameFast

        python_game = NFLGame(single_play=True, use_simple_model=True)
        if outcome_model is not None:
            python_game.cached_model = outcome_model
        if special_teams is not None:
            python_game.special_teams = special_teams

        cython_game = NFLGameFast(
            single_play=True,
            outcome_model=outcome_model,
            special_teams=special_teams,
            seed=123,
        )
        return python_game, cython_game

    def _assert_parity(self, python_game, cython_game):
        self.assertEqual(python_game.down, cython_game.down)
        self.assertEqual(python_game.ydstogo, cython_game.ydstogo)
        self.assertEqual(python_game.yardline, cython_game.yardline)
        self.assertEqual(python_game.phase, cython_game.phase)
        self.assertEqual(python_game.current_player, cython_game.current_player)
        self.assertEqual(python_game.is_over(), cython_game.is_over())
        py_payoffs = python_game.get_payoffs()
        cy_payoffs = cython_game.get_payoffs()
        self.assertAlmostEqual(py_payoffs[0], cy_payoffs[0], places=6)
        self.assertAlmostEqual(py_payoffs[1], cy_payoffs[1], places=6)

    def _run_play(self, python_game, cython_game, down, ydstogo, yardline, actions):
        python_game.init_game()
        cython_game.init_game()

        python_game.down = down
        python_game.ydstogo = ydstogo
        python_game.yardline = yardline

        cython_game.down = down
        cython_game.ydstogo = ydstogo
        cython_game.yardline = yardline

        for action in actions:
            python_game.step(action)
            cython_game.step(action)

    def test_first_down_parity(self):
        outcome_model = self._make_outcome_model(yards=10, turnover=False)
        python_game, cython_game = self._make_games(outcome_model=outcome_model)

        self._run_play(python_game, cython_game, down=1, ydstogo=10, yardline=25, actions=[0, 2, 0])

        self._assert_parity(python_game, cython_game)

    def test_turnover_parity(self):
        outcome_model = self._make_outcome_model(yards=3, turnover=True)
        python_game, cython_game = self._make_games(outcome_model=outcome_model)

        self._run_play(python_game, cython_game, down=2, ydstogo=7, yardline=40, actions=[1, 3, 1])

        self._assert_parity(python_game, cython_game)

    def test_touchdown_parity(self):
        outcome_model = self._make_outcome_model(yards=10, turnover=False)
        python_game, cython_game = self._make_games(outcome_model=outcome_model)

        self._run_play(python_game, cython_game, down=1, ydstogo=5, yardline=95, actions=[2, 1, 0])

        self._assert_parity(python_game, cython_game)

    def test_turnover_on_downs_parity(self):
        outcome_model = self._make_outcome_model(yards=2, turnover=False)
        python_game, cython_game = self._make_games(outcome_model=outcome_model)

        self._run_play(python_game, cython_game, down=4, ydstogo=5, yardline=60, actions=[3, 0, 1])

        self._assert_parity(python_game, cython_game)

    def test_safety_parity(self):
        outcome_model = self._make_outcome_model(yards=-5, turnover=False)
        python_game, cython_game = self._make_games(outcome_model=outcome_model)

        self._run_play(python_game, cython_game, down=2, ydstogo=10, yardline=2, actions=[4, 4, 1])

        self._assert_parity(python_game, cython_game)

    def test_special_teams_fg_parity(self):
        special_teams = self._make_special_teams(fg_prob=1.0)
        python_game, cython_game = self._make_games(special_teams=special_teams)

        self._run_play(python_game, cython_game, down=4, ydstogo=7, yardline=70, actions=[6])

        self._assert_parity(python_game, cython_game)

    def test_special_teams_punt_parity(self):
        special_teams = self._make_special_teams(punt_yardline=55)
        python_game, cython_game = self._make_games(special_teams=special_teams)

        self._run_play(python_game, cython_game, down=4, ydstogo=10, yardline=45, actions=[5])

        self._assert_parity(python_game, cython_game)


if __name__ == "__main__":
    unittest.main()
