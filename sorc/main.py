"""
Main file for everything - importing first 
"""
import lib

spotify = lib.reader("spotify.csv")

mean = lib.mean("duration_ms", spotify)
median = lib.median("duration_ms", spotify)
mode = lib.mode("duration_ms", spotify)
std = lib.std("duration_ms", spotify)
x = lib.viz("artist_name", spotify) # creates visualization and saves it in the output folder
