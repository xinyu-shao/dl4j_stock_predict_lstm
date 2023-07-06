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

object train_model {
  def main(args:Array[String]): Unit ={
    val basePath = new File("src/main/resources/data/")
    val trainingFiles = new File(basePath, "train/")
    val testFiles = new File(basePath, "test/")

    val trainSize = 1271
    val testSize = 544
    val miniBatchSize = 1
    val numPossibleLabels = -1
    val labelIndex = 0
    val regression = true

    // Training Data
    val reader = new CSVSequenceRecordReader()
    reader.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath}/%d.csv", 1000, trainSize))

    val trainData = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, regression)

    //Normalize the training data
    val normalizer = new NormalizerStandardize()
    normalizer.fitLabel(true)
    normalizer.fit(trainData) //Collect training data statistics
    trainData.reset()

    normalizer.save(
      new File(basePath.getAbsolutePath + "/model/a"),
      new File(basePath.getAbsolutePath + "/model/b"),
      new File(basePath.getAbsolutePath + "/model/c"),
      new File(basePath.getAbsolutePath + "/model/d")
    )

    trainData.setPreProcessor(normalizer)
    // Test Data
    val reader1 = new CSVSequenceRecordReader()
    reader1.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath}/%d.csv", 0, testSize-500))

    val testData = new SequenceRecordReaderDataSetIterator(reader1, miniBatchSize, numPossibleLabels, labelIndex, regression)

    testData.setPreProcessor(normalizer)

    val net = new lstm().MultiLayerNetwork()

    val uiServer = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    net.setListeners(new StatsListener(statsStorage))

    net.addListeners(new ScoreIterationListener(100))

    val nEpochs = 2000
    for (i <- 0 to nEpochs) {
      net.fit(trainData)
      //Evaluate on the test set:
      val evaluation: RegressionEvaluation = net.evaluateRegression(testData)
      println(s"======== EPOCH ${i} ========")
      println(evaluation.stats())

      trainData.reset()
      testData.reset()

      net.rnnClearPreviousState()
      if (i % 100 == 0) {

        var predicts: Array[Double] = Array()
        var actuals: Array[Double] = Array()

        while (trainData.hasNext && i % 50 == 0) {
          val nextTestPoint = trainData.next
          val nextTestPointFeatures = nextTestPoint.getFeatures
          val predictionNextTestPoint = net.output(nextTestPointFeatures)

          val nextTestPointLabels = nextTestPoint.getLabels
          normalizer.revert(nextTestPoint)
          normalizer.revertLabels(predictionNextTestPoint)
          println(s"Test point no.: ${nextTestPointFeatures} \n" +
            s"Prediction is: ${predictionNextTestPoint} \n" +
            s"Actual value is: ${nextTestPointLabels} \n")
          predicts = predicts :+ predictionNextTestPoint.getDouble(0L)
          actuals = actuals :+ nextTestPointLabels.getDouble(0L)
        }

        // visualize  predictions
        while (testData.hasNext) {
          val nextTestPoint = testData.next
          val nextTestPointFeatures = nextTestPoint.getFeatures

          val predictionNextTestPoint = net.output(nextTestPointFeatures) //net.rnnTimeStep(nextTestPointFeatures) // net.output(nextTestPointFeatures)

          val nextTestPointLabels = nextTestPoint.getLabels
          normalizer.revert(nextTestPoint) // revert the normalization of this test point
          normalizer.revertLabels(predictionNextTestPoint)
          predicts = predicts :+ predictionNextTestPoint.getDouble(0L)
          actuals = actuals :+ nextTestPointLabels.getDouble(0L)
        }
        PlotUtil.plot(predicts, actuals, s"Test Run", i)
        net.save(new File(basePath.getAbsolutePath+"/model/"+i))
        testData.reset()
        trainData.reset()

      }
    }
  }
}
