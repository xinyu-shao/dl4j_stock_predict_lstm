package predict

import lstm.plotting.PlotUtil
import org.datavec.api.records.reader.impl.csv._
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.deeplearning4j.datasets.datavec.{RecordReaderDataSetIterator, SequenceRecordReaderDataSetIterator}
import model.lstm
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.{NormalizerMinMaxScaler, NormalizerStandardize}

import java.io.File

object predict {
  def main(args:Array[String]): Unit ={
    val basePath = new File("src/main/resources/data/")
    val trainingFiles = new File(basePath, "train/")
    val testFiles = new File(basePath, "test/")

    val trainSize = 11
    val testSize = 4
    val miniBatchSize = 1
    val numPossibleLables = -1
    val regression = true

    // Training Data
    val trainFeatures = new CSVSequenceRecordReader
    trainFeatures.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath + "/train_Feature/"}%d.csv", 0, trainSize - 1))
    val trainLabels = new CSVSequenceRecordReader
    trainLabels.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath + "/train_label/"}%d.csv", 0, trainSize - 1))
    val trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numPossibleLables, regression)

    //Normalize the training data
    val normalizer = new NormalizerMinMaxScaler(0, 1)
    normalizer.fitLabel(true)
    normalizer.fit(trainDataIter) //Collect training data statistics
    trainDataIter.reset()

    // ----- Load the test data -----
    //Same process as for the training data.
    val testFeatures = new CSVSequenceRecordReader
    testFeatures.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath + "/test_Feature/"}%d.csv", 0, testSize - 1))
    val testLabels = new CSVSequenceRecordReader
    testLabels.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath + "/test_label/"}%d.csv", 0, testSize - 1))

    val testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true)

    trainDataIter.setPreProcessor(normalizer)
    testDataIter.setPreProcessor(normalizer)

    val net = new lstm().MultiLayerNetwork()

    //Initialize the user interface backend
    val uiServer = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    net.setListeners(new StatsListener(statsStorage))

    net.addListeners(new ScoreIterationListener(100))

    val nEpochs = 1
    for (i <- 0 to nEpochs) {
      net.fit(trainDataIter)
      //Evaluate on the test set:
      val evaluation: RegressionEvaluation = net.evaluateRegression(testDataIter)
      println(s"======== EPOCH ${i} ========")
      println(evaluation.stats())
      trainDataIter.reset()
      testDataIter.reset()

      var predicts: Array[Double] = Array()
      var actuals: Array[Double] = Array()

      while (trainDataIter.hasNext) {
        val nextTestPoint = trainDataIter.next
        val nextTestPointFeatures = nextTestPoint.getFeatures
        val predictionNextTestPoint: INDArray = net.output(nextTestPointFeatures) //net.rnnTimeStep(nextTestPointFeatures) // net.output(nextTestPointFeatures)

        val nextTestPointLabels = nextTestPoint.getLabels
        normalizer.revert(nextTestPoint) // revert the normalization of this test point
        println(s"Test point no.: ${nextTestPointFeatures} \n" +
          s"Prediction is: ${predictionNextTestPoint} \n" +
          s"Actual value is: ${nextTestPointLabels} \n")
        predicts = predicts :+ predictionNextTestPoint.getDouble(6L)
        actuals = actuals :+ nextTestPointLabels.getDouble(6L)
      }

      // visualize test data / predictions
      while (testDataIter.hasNext) {
        val nextTestPoint = testDataIter.next
        val nextTestPointFeatures = nextTestPoint.getFeatures
        val predictionNextTestPoint: INDArray = net.output(nextTestPointFeatures, true) //net.rnnTimeStep(nextTestPointFeatures) // net.output(nextTestPointFeatures)

        val nextTestPointLabels = nextTestPoint.getLabels
        normalizer.revert(nextTestPoint) // revert the normalization of this test point
        predicts = predicts :+ predictionNextTestPoint.getDouble(6L)
        actuals = actuals :+ nextTestPointLabels.getDouble(6L)
      }
      if (i % 10 == 0)
        PlotUtil.plot(predicts, actuals, s"Test Run", i)

      testDataIter.reset()
      trainDataIter.reset()
      net.rnnClearPreviousState()
    }
  }
}
