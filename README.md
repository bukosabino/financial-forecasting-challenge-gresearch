# Financial Forecasting Challenge G-Research

This repository include my code to the challenge proposed by G-Research:

https://financialforecasting.gresearch.co.uk/

In ended up at 29th place on the private leaderboard, among about 400 participants.

You can read my notes about the challenge:

https://medium.com/@bukosabino/financial-forecasting-challenge-by-g-research-8792c5344ae9
# Deployment

```sh
$ virtualenv -p python3 env
$ source env/bin/activate
$ pip install -r requirements.txt
```

# Run

0) You need to download the train and test datasets of the challenge: https://financialforecasting.gresearch.co.uk/
And put them in a 'input' folder in the project.

1) You need to execute preprocessing script. So:

```sh
$ cd preprocessing
$ python preprocessing.py
```

2) You can use any model in model folder. So:

```sh
$ cd models
$ python modelX.py
```

# Credits

Developed by Bukosabino at Lecrin Technologies - http://lecrintech.com

We are a cooperative of freelancers focused on Data Science. Please, let me know about any comment or project related.
