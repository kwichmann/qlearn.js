// This demonstrates a random walk between 5 state.
// State 0 and 4 are terminal, with rewards -1 and 1 respectively.

function simulate() {
    let episode = new Episode();

    let state = 2;
    let action;
    let reward = 0;
    let episodeEnded = false;

    while(!episodeEnded) {
        action = Math.random() < 0.5 ? 0 : 1;

        if (state == 0) {
            reward = -1;
            action = null;
            episodeEnded = true;
        }
        if (state == 4) {
            reward = 1;
            action = null;
            episodeEnded = true;
        }
        episode.add([state], [action], reward);

        if (action == 0) {
            state -= 1;
        } else {
            state += 1;
        }
    }

    return episode;
}

let buffer = new ReplayBuffer(100, [1], [1]);

for (let i = 0; i < 10; i++) {
    let episode = simulate();
    buffer.addEpisode(episode, 1);
}

const q = tf.variable(tf.zeros([5, 2]));

const inputState = tf.input({shape: [1]});
const inputAction = tf.input({shape: [1]});

const concatLayer = tf.layers.concatenate();
const input = concatLayer.apply([inputState, inputAction]);

const hidden = tf.layers.dense({units: 10, activation: "relu"}).apply(input);
const output = tf.layers.dense({units: 1, activation: "linear"}).apply(hidden);

const approx = tf.model({inputs: [inputState, inputAction], outputs: output});

let model = new Qmodel(approx);
 
const opt = tf.train.sgd(0.1);

model.setTrainingParameters(
    {
        type: "MC",
        gamma: 1
    },
    opt
);

model.train(buffer, 100).then(() => {console.log("Training done");});