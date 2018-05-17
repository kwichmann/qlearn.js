// Class for keeping track of an episode for MC learning
class Episode {
    constructor() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
    }

    add(state, action, reward) {
        this.states.push(state);
        this.actions.push(action);
        this.rewards.push(reward);
    }
}

const flatten = (arr) => arr.reduce((flat, next) => flat.concat(Array.isArray(next) ? flatten(next) : next), []);

// Class for experience replay data
class ReplayBuffer {
    constructor(bufferSize = 1000,          // How many training examples to keep in active memory?
        stateShape,
        actionShape)                    
    {
        this.stateShape = stateShape;
        this.stateSize = stateShape.reduce((a, b) => a * b, 1);
        
        this.actionShape = actionShape;
        this.actionSize = actionShape.reduce((a, b) => a * b, 1);
        
        this.stateBuffer = tf.buffer([bufferSize, this.stateSize]);
        this.actionBuffer = tf.buffer([bufferSize, this.actionSize]);
        this.targetBuffer = tf.buffer([bufferSize, 1]);
        
        this.bufferSize = bufferSize;
        this.currentSize = 0;
        this.currentPosition = 0;
    }

    add(state, action, target) {                  // Add a single training example to buffer
        flatten(state).map(
            (value, n) => {
                this.stateBuffer.set(value, this.currentPosition, n);
            }
        );
        
        flatten(action).map(
            (value, n) => {
                this.actionBuffer.set(value, this.currentPosition, n);
            }
        );
        
        this.targetBuffer.set(target, this.currentPosition, 0);
        
        this.currentPosition += 1;
        if (this.currentPosition >= this.bufferSize) {
            this.currentPosition = 0;
        }

        this.currentSize += 1;
        if (this.currentSize >= this.bufferSize) {
            this.currentSize = this.bufferSize;
        }
    }

    addEpisode(episode, gamma) {
        const l = episode.states.length;
        
        let target = episode.rewards[l - 1];
        
        for (let i = l - 2; i >= 0; i--) {
            const state = episode.states[i];
            const action = episode.actions[i];
            const reward = episode.rewards[i];
            
            target = reward + target * gamma;

            this.add(state, action, target);
        }
    }

    getStates() {
        return this.stateBuffer.toTensor()
                .slice([0, 0], [this.currentSize, this.stateSize])
                .reshape([this.currentSize].concat(this.stateShape))
    }

    getActions() {
        return this.actionBuffer.toTensor()
                .slice([0, 0], [this.currentSize, this.actionSize])
                .reshape([this.currentSize].concat(this.actionShape))
    }

    getTargets() {
        return this.targetBuffer.toTensor()
                .slice([0, 0], [this.currentSize, 1])
    }
}

class Qmodel {
    constructor(approximationFunction)      // Action value function approximator tf.model
    {    
        this.approximator = approximationFunction;
    }

    setTrainingParameters(algorithm,        // Object with 'type', 'gamma', and possibly 'lambda'
        optimizer,                          // A TensorFlow optimizer function
        fixedTargetDelay = 1,               // How often should the target q be updated?
        miniBatchSize = 32,                 // Size of minibatches
        prematureTraining = false)          // Train before miniBatchSize is reached?)
    {
        this.algorithm = algorithm["type"];
        this.gamma = algorithm["gamma"];
        
        switch(this.algorithm) {
            case "MC":
            
            this.approximator.compile({
                optimizer: optimizer,
                loss: "meanSquaredError"
            });

            break;

            case "TD":
                this.lambda = algorithm["lambda"];
                this.eligibilityTrace = tf.zeros(inputShape);
                break;
            
            case "Sarsa":
                this.lambda = algorithm["lambda"];
                this.eligibilityTrace = tf.zeros(inputShape);
                break;
            
            case "Q-learning":
                this.lambda = algorithm["lambda"];
                break;

            default:
                console.log("Unknown algorithm")
                return;
            
            this.fixedTargetDelay = fixedTargetDelay;
            this.fixedTargetCounter = 0;
            this.miniBatchSize = miniBatchSize;
            this.prematureTraining = prematureTraining;
        }
    }

    train(replayBuffer, epochs = 1)                     // Buffer to train from
    {
        tf.tidy(() => {
            const states = replayBuffer.getStates();
            const actions = replayBuffer.getActions();
            const targets = replayBuffer.getTargets();

            this.approximator.fit([states, actions], targets, {
                batchSize: this.miniBatchSize,
                epochs: epochs
            });
        });
    }
}
