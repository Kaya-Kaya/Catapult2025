## Model training guidelines

Videos -> Model workflow

1. Use extract_frames.py to extract the frames. Specify paths according to data. Input videos stored in data/SIDE VIEW. Output frames stored in folders in output_frames/SIDE VIEW (1 folder for each video).

2. Use extract_features.py to extract features from the frames. Specify paths and features according to what data/element you're trying to analyse.

3. Train the model on the data. Change values and labels appropriately. 