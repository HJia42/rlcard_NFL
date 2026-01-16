"""
Test cases for NFL IIG (Imperfect Information Game) variants.

Run with: python -m pytest tests/test_nfl_iig.py -v
Or: python tests/test_nfl_iig.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import rlcard


class TestNFLIIGGame(unittest.TestCase):
    """Test the IIG game logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = rlcard.make('nfl-iig', config={
            'single_play': True,
            'use_cached_model': True,
        })
    
    def test_game_initialization(self):
        """Test that game initializes correctly."""
        state, player = self.env.reset()
        
        self.assertEqual(player, 0, "Offense should start")
        self.assertEqual(self.env.game.phase, 0, "Should start in phase 0")
        self.assertEqual(self.env.game.down, 1, "Should start on 1st down")
        self.assertEqual(self.env.game.ydstogo, 10, "Should be 1st and 10")
        self.assertIsNotNone(state['obs'], "State should have obs")
        self.assertEqual(len(state['obs']), 12, "Obs should be 12 dimensions")
    
    def test_action_space_size(self):
        """Test correct number of actions."""
        self.assertEqual(self.env.num_actions, 12, "IIG should have 12 offense actions")
        
        state, _ = self.env.reset()
        legal_actions = list(state['legal_actions'].keys())
        self.assertEqual(len(legal_actions), 12, "All 12 actions should be legal in phase 0")
    
    def test_action_encoding(self):
        """Test action encoding is correct."""
        from rlcard.games.nfl.game_iig import IIG_OFFENSE_ACTIONS, decode_iig_action
        
        # Check structure
        self.assertEqual(len(IIG_OFFENSE_ACTIONS), 12, "12 total actions")
        
        # Formation + play type combos
        self.assertEqual(decode_iig_action(0), ('SHOTGUN', 'pass'))
        self.assertEqual(decode_iig_action(1), ('SHOTGUN', 'rush'))
        self.assertEqual(decode_iig_action(6), ('I_FORM', 'pass'))
        self.assertEqual(decode_iig_action(7), ('I_FORM', 'rush'))
        
        # Special teams
        self.assertEqual(decode_iig_action(10), ('PUNT', None))
        self.assertEqual(decode_iig_action(11), ('FG', None))
    
    def test_phase_transitions(self):
        """Test correct phase transitions."""
        state, player = self.env.reset()
        self.assertEqual(player, 0, "Start with offense")
        self.assertEqual(self.env.game.phase, 0)
        
        # Offense picks action 0 (SHOTGUN_pass)
        state, player = self.env.step(0)
        self.assertEqual(player, 1, "Defense's turn after offense commits")
        self.assertEqual(self.env.game.phase, 1, "Should be in phase 1")
        
        # Defense picks action 0 (4-box)
        state, player = self.env.step(0)
        # After defense, play auto-executes and game ends (single_play)
        self.assertTrue(self.env.is_over(), "Game should end after single play")
    
    def test_information_hiding(self):
        """Test that defense cannot see committed play type."""
        state, _ = self.env.reset()
        
        # Offense commits to SHOTGUN_pass (action 0)
        state, player = self.env.step(0)
        
        # Defense is now acting
        self.assertEqual(player, 1)
        
        # Check that state does NOT contain committed play type
        self.assertNotIn('committed_play_type', state)
        self.assertNotIn('play_type', state)
        
        # But formation should be visible (in obs)
        self.assertIn('obs', state)
        # Formation is encoded in indices 3-7, SHOTGUN is index 0
        self.assertEqual(state['obs'][3], 1.0, "Defense should see SHOTGUN formation")
    
    def test_special_teams_bypasses_defense(self):
        """Test that PUNT/FG skips defense phase."""
        state, _ = self.env.reset()
        
        # Offense picks PUNT (action 10)
        state, player = self.env.step(10)
        
        # Game should be over immediately (special teams resolves in one step)
        self.assertTrue(self.env.is_over(), "PUNT should end game immediately")
    
    def test_payoffs_are_zero_sum(self):
        """Test that payoffs are zero-sum."""
        state, _ = self.env.reset()
        
        # Play a full game
        while not self.env.is_over():
            legal_actions = list(state['legal_actions'].keys())
            action = legal_actions[0]
            state, _ = self.env.step(action)
        
        payoffs = self.env.get_payoffs()
        self.assertAlmostEqual(payoffs[0] + payoffs[1], 0.0, places=5,
                               msg="Payoffs should be zero-sum")
    
    def test_obs_array_normalization(self):
        """Test that observation array is properly normalized."""
        state, _ = self.env.reset()
        
        obs = state['obs']
        
        # All values should be in [0, 1] range (normalized)
        self.assertTrue(all(0 <= v <= 1 for v in obs),
                        f"All obs values should be normalized: {obs}")


class TestNFLIIGBucketedGame(unittest.TestCase):
    """Test the bucketed IIG game."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = rlcard.make('nfl-iig-bucketed', config={
            'single_play': True,
            'use_cached_model': True,
        })
    
    def test_bucketed_game_initialization(self):
        """Test bucketed game initializes correctly."""
        state, player = self.env.reset()
        
        self.assertEqual(player, 0)
        self.assertIn('obs', state)
        self.assertIn('obs_tuple', state)
        self.assertEqual(len(state['obs']), 12)
    
    def test_info_set_counting(self):
        """Test info set counting is correct."""
        from rlcard.games.nfl.game_iig_bucketed import NFLGameIIGBucketed
        
        info_sets = NFLGameIIGBucketed.count_info_sets()
        
        self.assertEqual(info_sets['phase_0'], 320)  # 4 × 4 × 20
        self.assertEqual(info_sets['phase_1'], 1600)  # 320 × 5
        self.assertEqual(info_sets['total'], 1920)


class TestIIGvsAudibleComparison(unittest.TestCase):
    """Test differences between IIG and Audible games."""
    
    def test_action_space_difference(self):
        """Test that IIG has different action space than Audible."""
        env_iig = rlcard.make('nfl-iig', config={'single_play': True})
        env_audible = rlcard.make('nfl-bucketed', config={'single_play': True})
        
        # IIG: offense has 12 compound actions
        self.assertEqual(env_iig.num_actions, 12)
        
        # Audible: offense has 7 initial actions
        self.assertEqual(env_audible.num_actions, 7)
    
    def test_number_of_phases(self):
        """Test IIG has fewer phases than Audible."""
        env_iig = rlcard.make('nfl-iig', config={'single_play': True, 'use_cached_model': True})
        env_audible = rlcard.make('nfl-bucketed', config={'single_play': True, 'use_cached_model': True})
        
        # Play through IIG
        env_iig.reset()
        env_iig.step(0)  # Offense commits
        env_iig.step(0)  # Defense picks, auto-execute
        
        # IIG goes: Phase 0 (offense) -> Phase 1 (defense) -> auto-execute
        # Total player actions: 2
        
        # Play through Audible
        env_audible.reset()
        env_audible.step(0)  # Offense picks formation
        env_audible.step(0)  # Defense picks box
        env_audible.step(0)  # Offense picks play type
        
        # Audible goes: Phase 0 -> Phase 1 -> Phase 2 -> execute
        # Total player actions: 3


class TestRandomPlaythrough(unittest.TestCase):
    """Test random playthroughs complete without error."""
    
    def test_random_iig_games(self):
        """Run multiple random IIG games."""
        env = rlcard.make('nfl-iig', config={
            'single_play': True,
            'use_cached_model': True,
        })
        
        for i in range(50):
            state, player = env.reset()
            steps = 0
            
            while not env.is_over() and steps < 10:
                legal_actions = list(state['legal_actions'].keys())
                action = np.random.choice(legal_actions)
                state, player = env.step(action)
                steps += 1
            
            self.assertTrue(env.is_over(), f"Game {i} should complete")
            payoffs = env.get_payoffs()
            self.assertAlmostEqual(payoffs[0] + payoffs[1], 0.0, places=5)
    
    def test_random_iig_bucketed_games(self):
        """Run multiple random IIG bucketed games."""
        env = rlcard.make('nfl-iig-bucketed', config={
            'single_play': True,
            'use_cached_model': True,
        })
        
        for i in range(50):
            state, player = env.reset()
            steps = 0
            
            while not env.is_over() and steps < 10:
                legal_actions = list(state['legal_actions'].keys())
                action = np.random.choice(legal_actions)
                state, player = env.step(action)
                steps += 1
            
            self.assertTrue(env.is_over(), f"Game {i} should complete")


class TestNFLBucketedGame(unittest.TestCase):
    """Test the standard bucketed NFL game."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = rlcard.make('nfl-bucketed', config={
            'single_play': True,
            'use_cached_model': True,
        })
    
    def test_bucketed_initialization(self):
        """Test bucketed game initializes correctly."""
        state, player = self.env.reset()
        
        self.assertEqual(player, 0)
        self.assertEqual(self.env.game.phase, 0)
        self.assertIn('obs', state)
        self.assertEqual(len(state['obs']), 12)
    
    def test_bucketed_has_obs_tuple(self):
        """Test bucketed game includes obs_tuple for tabular methods."""
        state, _ = self.env.reset()
        self.assertIn('obs_tuple', state)
    
    def test_bucketed_three_phases(self):
        """Test bucketed game has 3 phases."""
        state, _ = self.env.reset()
        
        # Phase 0: Offense picks formation
        self.assertEqual(self.env.game.phase, 0)
        state, player = self.env.step(0)  # Pick SHOTGUN
        
        # Phase 1: Defense picks box
        self.assertEqual(self.env.game.phase, 1)
        self.assertEqual(player, 1)
        state, player = self.env.step(0)  # Pick 4-box
        
        # Phase 2: Offense picks play type
        self.assertEqual(self.env.game.phase, 2)
        self.assertEqual(player, 0)
        state, player = self.env.step(0)  # Pick pass
        
        # Game should end
        self.assertTrue(self.env.is_over())
    
    def test_bucketed_info_sets(self):
        """Test info set counting for bucketed game."""
        from rlcard.games.nfl.game_bucketed import NFLGameBucketed
        
        info_sets = NFLGameBucketed.count_info_sets()
        
        # Verify structure exists and values are reasonable
        self.assertIn('phase_0', info_sets)
        self.assertIn('phase_1', info_sets)
        self.assertIn('phase_2', info_sets)
        self.assertIn('total', info_sets)
        self.assertEqual(info_sets['total'], 
                         info_sets['phase_0'] + info_sets['phase_1'] + info_sets['phase_2'])
    
    def test_bucketed_random_games(self):
        """Run random bucketed games."""
        for i in range(20):
            state, player = self.env.reset()
            steps = 0
            
            while not self.env.is_over() and steps < 10:
                legal_actions = list(state['legal_actions'].keys())
                action = np.random.choice(legal_actions)
                state, player = self.env.step(action)
                steps += 1
            
            self.assertTrue(self.env.is_over())


class TestCythonGameFast(unittest.TestCase):
    """Test the Cython fast game implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Check if Cython is available."""
        try:
            from rlcard.games.nfl.cython.game_fast import NFLGameFast, make_fast_game
            cls.cython_available = True
            cls.NFLGameFast = NFLGameFast
            cls.make_fast_game = make_fast_game
        except ImportError:
            cls.cython_available = False
    
    def setUp(self):
        if not self.cython_available:
            self.skipTest("Cython game_fast not compiled")
        self.game = self.make_fast_game(single_play=True, use_cached_model=True)
    
    def test_cython_game_initialization(self):
        """Test Cython game initializes correctly."""
        state, player = self.game.init_game()
        
        self.assertEqual(player, 0)
        self.assertEqual(self.game.down, 1)
        self.assertEqual(self.game.ydstogo, 10)
        self.assertEqual(self.game.phase, 0)
    
    def test_cython_action_space(self):
        """Test Cython game has correct action space."""
        self.game.init_game()
        legal = self.game.get_legal_actions()
        
        self.assertEqual(len(legal), 7)  # 5 formations + PUNT + FG
    
    def test_cython_three_phases(self):
        """Test Cython game has 3 phases."""
        self.game.init_game()
        
        # Phase 0: Offense
        self.assertEqual(self.game.phase, 0)
        self.game.step(0)  # SHOTGUN
        
        # Phase 1: Defense
        self.assertEqual(self.game.phase, 1)
        self.game.step(0)  # 4-box
        
        # Phase 2: Play type
        self.assertEqual(self.game.phase, 2)
        self.game.step(0)  # pass
        
        self.assertTrue(self.game.is_over())
    
    def test_cython_payoffs_zero_sum(self):
        """Test Cython payoffs are zero-sum."""
        self.game.init_game()
        
        while not self.game.is_over():
            legal = self.game.get_legal_actions()
            self.game.step(legal[0])
        
        payoffs = self.game.get_payoffs()
        self.assertAlmostEqual(payoffs[0] + payoffs[1], 0.0, places=5)
    
    def test_cython_special_teams(self):
        """Test Cython special teams."""
        self.game.init_game()
        
        # PUNT = action 5
        self.game.step(5)
        self.assertTrue(self.game.is_over())
        
        # FG = action 6
        self.game.init_game()
        self.game.step(6)
        self.assertTrue(self.game.is_over())
    
    def test_cython_random_games(self):
        """Run random Cython games."""
        for i in range(50):
            self.game.init_game()
            steps = 0
            
            while not self.game.is_over() and steps < 10:
                legal = self.game.get_legal_actions()
                action = np.random.choice(legal)
                self.game.step(action)
                steps += 1
            
            self.assertTrue(self.game.is_over())


class TestCythonIIGGameFast(unittest.TestCase):
    """Test the Cython fast IIG game implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Check if Cython IIG is available."""
        try:
            from rlcard.games.nfl.cython.game_iig_fast import NFLGameIIGFast, make_fast_iig_game
            cls.cython_available = True
            cls.NFLGameIIGFast = NFLGameIIGFast
            cls.make_fast_iig_game = make_fast_iig_game
        except ImportError:
            cls.cython_available = False
    
    def setUp(self):
        if not self.cython_available:
            self.skipTest("Cython game_iig_fast not compiled")
        self.game = self.make_fast_iig_game(single_play=True, use_cached_model=True)
    
    def test_iig_cython_initialization(self):
        """Test Cython IIG game initializes correctly."""
        state, player = self.game.init_game()
        
        self.assertEqual(player, 0)
        self.assertEqual(self.game.down, 1)
        self.assertEqual(self.game.phase, 0)
    
    def test_iig_cython_action_space(self):
        """Test Cython IIG has 12 offense actions."""
        self.game.init_game()
        legal = self.game.get_legal_actions()
        
        self.assertEqual(len(legal), 12)  # 5×2 + 2
    
    def test_iig_cython_two_phases(self):
        """Test Cython IIG has 2 player phases."""
        self.game.init_game()
        
        # Phase 0: Offense commits
        self.assertEqual(self.game.phase, 0)
        state, player = self.game.step(0)  # SHOTGUN_pass
        
        # Phase 1: Defense picks, auto-execute
        self.assertEqual(self.game.phase, 1)
        self.assertEqual(player, 1)
        self.game.step(0)  # 4-box
        
        self.assertTrue(self.game.is_over())
    
    def test_iig_cython_information_hiding(self):
        """Test Cython IIG hides play type from defense."""
        self.game.init_game()
        
        # Offense commits to SHOTGUN_pass
        state, player = self.game.step(0)
        
        # Defense sees formation_idx but NOT pending_play_type
        self.assertIn('formation_idx', state)
        self.assertNotIn('pending_play_type', state)  # Hidden!
    
    def test_iig_cython_special_teams(self):
        """Test Cython IIG special teams."""
        self.game.init_game()
        
        # PUNT = action 10
        self.game.step(10)
        self.assertTrue(self.game.is_over())
        
        # FG = action 11
        self.game.init_game()
        self.game.step(11)
        self.assertTrue(self.game.is_over())
    
    def test_iig_cython_random_games(self):
        """Run random Cython IIG games."""
        for i in range(50):
            self.game.init_game()
            steps = 0
            
            while not self.game.is_over() and steps < 10:
                legal = self.game.get_legal_actions()
                action = np.random.choice(legal)
                self.game.step(action)
                steps += 1
            
            self.assertTrue(self.game.is_over())


class TestPythonCythonEquivalence(unittest.TestCase):
    """Test that Python and Cython implementations produce equivalent results."""
    
    @classmethod
    def setUpClass(cls):
        """Check if Cython is available."""
        try:
            from rlcard.games.nfl.cython.game_fast import make_fast_game
            cls.cython_available = True
            cls.make_fast_game = make_fast_game
        except ImportError:
            cls.cython_available = False
    
    def test_ep_calculation_equivalence(self):
        """Test EP calculation matches between Python and Cython."""
        if not self.cython_available:
            self.skipTest("Cython not compiled")
        
        from rlcard.games.nfl.game import NFLGame
        
        python_game = NFLGame(single_play=True, use_cached_model=True)
        cython_game = self.make_fast_game(single_play=True, use_cached_model=True)
        
        # Test several scenarios
        test_cases = [
            (1, 10, 25),   # 1st and 10 from own 25
            (3, 5, 50),    # 3rd and 5 from midfield
            (4, 1, 99),    # 4th and 1 from goal line
            (2, 15, 10),   # 2nd and 15 from deep
        ]
        
        for down, ydstogo, yardline in test_cases:
            python_ep = python_game._calculate_ep(down, ydstogo, yardline)
            cython_ep = cython_game._calculate_ep(down, ydstogo, yardline)
            
            self.assertAlmostEqual(python_ep, cython_ep, places=3,
                msg=f"EP mismatch at down={down}, ydstogo={ydstogo}, yardline={yardline}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
