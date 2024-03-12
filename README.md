This project aims to analyze a dataset of events generated from football matches of the 5 major European leagues of the 2017/2018 season, the 2016 European Championship and the 2018 World Cup.

The first objective of the analysis was to study the distances covered in a football match, as well as to understand how they can be compared with the distances of human mobility.

The second objective of the project was to examine and understand how the length of passing chains can increase or decrease the probability of taking a shot, scoring a goal or winning the match.

The last part of the project focused on examining and calculating the predictability of passing chains performed during a match. The football field was divided into tiles and, through a recurrent neural network (RNN), an attempt was made to understand how predictable the passing chains were.

The project has been divided into 3 notebooks, each containing a specific task required by the project:

- Task 1 is present in the `Distances_analysis.ipynb` notebook.
- Task 2 is present in the `Pass_chains_analysis.ipynb` notebook.
- Task 3 is present in the `Pass_chains_predictability.ipynb` notebook.

In addition, there are two other files in the folder:

- `utils.py` contains functions used in the notebooks mentioned above.
- `requirements.txt` contains all the libraries, with their respective versions, that were used to run the code.

The dataset used for the project is available here: https://figshare.com/collections/Soccer_match_event_dataset/4415000/2.