// Class for keeping track of an episode for MC learning
class Episode {
    constructor() {
        this.samples = [];
    }

    add(sample) {
        this.samples.push(sample);
    }
}

// Class for experience replay data
class ReplayBuffer {
    constructor(bufferSize = 1000,          // How many training examples to keep in active memory?
        oneHotState = false)                // Represent states with a single input being 1?
    {
        this.buffer = [];
        this.bufferSize = bufferSize;
        this.oneHotState = oneHotState;
    }

    add(trainingExample) {                  // Add a single training example to buffer
        this.buffer.push(trainingExample);
        if (this.buffer.length > this.bufferSize) {
            this.buffer.shift();
        }
    }

    addEpisode(episode, gamma) {
        const l = episode.samples.length;
        let targetReward = episode.samples[l - 1]["r"];
        
        for (let i = l - 2; i >= 0; i--) {
            const sample = episode.samples[i];
            
            const state = sample["state"];
            const action = sample["action"];
            const r = sample["r"];
            
            targetReward = r + targetReward * gamma;
            
            this.add({
                state: state,
                action: action,
                target: targetReward
            });
        }
    }
}

class Qmodel {
    constructor(inputShape,                 // Shape of representation of state
        outputShape,                        // Shape of representation of action
        approximationFunction)              // Action value function approximator
    {    
        this.inputShape = inputShape;
        this.outputShape = outputShape;
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
            // For MC learning, we need to keep track of events until the episode ends
            this.episode = new Episode();

            // Make targets
            let target = (sample) => tf.scalar(sample["target"]);            
            let targets = (batch) => tf.tensor1d(batch.map(target));
            
            // Find current q-values
            let currentStateAction = (sample) => approximationFunction(sample["state"], sample["action"]);
            let currentQ = (batch) => tf.tensor1d(batch.map(currentStateAction));

            this.loss = (batch) => tf.tidy(tf.squaredDifference(targets(batch), currentQ(batch)).mean());
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
            
            this.optimizer = optimizer;
            this.fixedTargetDelay = fixedTargetDelay;
            this.fixedTargetCounter = 0;
            this.miniBatchSize = miniBatchSize;
            this.prematureTraining = prematureTraining;

            
        }
    }

    train(replayBuffer)                     // Buffer to train from
    {
        let numberOfTrainingExamples = this.replayBuffer.length;
        if (numberOfTrainingExamples == 0) {
            return;
        }
        if (numberOfTrainingExamples < this.replayBuffer.bufferSize) {
            if (prematureTraining) {
                return;
            }
        }
        this.optimizer.minimize()
    }
}
