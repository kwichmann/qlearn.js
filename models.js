class ExactModel {
    constructor(numStates, numActions) {
        const stateInput = tf.input({shape: [1]});
        const actionInput = tf.input({shape: [1]});
        
        const index = tf.layers.add().apply([tf.layers.multiply().apply([tf.scalar(numActions), actionInput]), stateInput]);
        
        const embed = tf.layers.embedding({
            inputDim: numStates * numInput,
            outputDim: 1,
            embeddingsInitializer: "zeros"
        }).apply(index);

        const model = tf.model({
            inputs: [stateInput, actionInput],
            outputs: embed
        });

        return new Qmodel(model);
    }
}