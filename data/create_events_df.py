def create_events_df(activityName, algorithm):
    import numpy as np
    import pandas as pd
    from utils.participant_info_utils import get_generic_trial_nums
    filenameList = []
    for subjectNum in range(1, 72):
        for trialNum in get_generic_trial_nums(subjectNum, [activityName]):
            opticalFilename = "TF_{:02d}_{:04d}.c3d".format(subjectNum, trialNum)
            filenameList.append(opticalFilename)

    df = pd.DataFrame(data=np.full((len(filenameList), 4), np.nan),
                      index=pd.Index(data=filenameList, name="Filename"),
                      columns=["Left ICs", "Right ICs", "Left FOs", "Right FOs"])
    df.to_csv("{}/Events/{}Events.csv".format(algorithm, activityName))


if __name__ == "__main__":
    for activityName in ["Walk", "WalkNod", "WalkShake", "TUG"]:
        for algorithm in ["Diao", "TP-EAR"]:
            create_events_df(activityName, algorithm)
