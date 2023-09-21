'''imports'''
import sys
sys.path.append('/workspaces/pythonCiCd_assignment1_fj49')
from sorc.lib import mean, median, mode, std, viz, reader

spotify = reader("spotify.csv")
column_ds = "duration_ms"
column_viz = "artist_name"
mean = mean(column_ds,spotify)
median = median(column_ds, spotify)
mode = mode(column_ds,spotify)
std = std(column_ds,spotify)
viz = viz(column_viz,spotify)

def test_mean():
    '''tests mean'''
    assert mean == int(spotify["duration_ms"].mean())


def test_median():
    '''tests median'''
    assert median == int(spotify["duration_ms"].median())


def test_mode():
    '''tests mode'''
    assert mode == int(spotify["duration_ms"].mode())


def test_std():
    '''tests std'''
    assert std == int(spotify["duration_ms"].std())


def test_viz():
    '''tests viz'''
    assert viz is not None  # asserts that viz is not empty