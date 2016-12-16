The environment we use is python 3. The code is in two files, TeamJ.py and TSP_2MST.py, where the MST approximate algorithm is in TSP_2MST.py and the rest is in TeamJ.py. To execute, simply command the following line:
python data_path cut_off_time method random_seed
for example: for branch and bound, command:
python3 TeamJ.py Atlanta.tsp 600 BnB None
for local search one, command:
python3 TeamJ.py Atlanta.tsp 10 LS1 1
for local search two, command:
python3 TeamJ.py Atlanta.tsp 10 LS2 1
for heuristic approximation, command:
python3 TeamJ.py Atlanta.tsp 10 Heur 1
for MST approximation, command:
python3 TeamJ.py Atlanta.tsp 10 MST 1

