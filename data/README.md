# Data

The repository does not redistribute the datasets used for the recorded runs.

The translation code was developed with the Vietnamese-English parallel data distributed through the course dataset repository (`data.30.vi` and `data.30.en`).

Aspect sentiment classification used the public MAMS-ATSA review dataset. The original data are available from the MAMS-for-ABSA repository.

Humour-score prediction used the English data from SemEval 2021 HaHackathon Task 1b. Ratings in the recorded experiments were divided by 5 before training, so the reported MSE values are on a 0-1 target scale.

T5 summarization used a 10% subset of the XSum validation split, divided again into train and validation sets for the experiment.
