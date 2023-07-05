package model

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

class lstm {
  def MultiLayerNetwork(): MultiLayerNetwork ={
    val seed = 12345
    val learningRate = 0.15
    val numInputs = 7
    val lstm1Size = 50
    val lstm2Size = 100
    val numOutputs = 1

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .activation(Activation.TANH)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(learningRate, 0.9))
      .list()
      .layer(0, new LSTM.Builder()
        .activation(Activation.TANH)
        .nIn(numInputs)
        .nOut(lstm1Size).build())
      .layer(1, new LSTM.Builder()
        .activation(Activation.TANH)
        .nIn(lstm1Size)
        .nOut(lstm2Size).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(lstm2Size)
        .nOut(numOutputs).build())
      .build();

    val net = new MultiLayerNetwork(conf)
    net.init()

    net
  }
}
