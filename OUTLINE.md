# Project Title

## Outline
Golf Assistant with the following workflow:
 - Video to frames
 - extract coordinates of relevant posture nodes from frames
    - (x,y) for every node
    - 1 node for every limb/body part/category
    - set of nodes for each frame
    - set of frames for each video
 - neural network:
    - input: set of frames, with set of nodes, with set of coordinates
    - i.e. [[(x, y)]] where (x,y) are the coordinates
    - output: vector with posture scores
    - i.e. [0.85, 0.97, 0.42]
    - along with a pre-established mapping of node name to node scores to make sense of it
    - i.e. {chin: 0.85, right_knee: 0.97, left_knee: 0.42, ...}
 - hashmap --- LLM ---> feedback:
    - prompt: motivational, constructive, etc.
    - word limit
    - suggested steps/exercises to improve