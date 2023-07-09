package another

import lstm.plotting.PlotUtil
import model.model
import org.datavec.api.records.reader.impl.csv._
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler

import java.io.File

object train_mutil_model {
  def main(args:Array[String]): Unit ={
    val basePath = new File("src/main/resources/data/")
    val trainingFiles = new File(basePath, "train/")
    val testFiles = new File(basePath, "test/")

    val trainSize = 1277
    val testSize = 545
    val miniBatchSize = 1
    val numPossibleLables = 10000000
    val regression = true

    // Training Data
    val trainFeatures = new CSVSequenceRecordReader
    trainFeatures.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath + "/train_Feature/"}%d.csv", 0, trainSize))
    val trainLabels = new CSVSequenceRecordReader
    trainLabels.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath + "/train_label/"}%d.csv", 0, 30))
    val trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numPossibleLables, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH)

    // ----- Load the test data -----
    //Same process as for the training data.
    val testFeatures = new CSVSequenceRecordReader
    testFeatures.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath + "/test_Feature/"}%d.csv", 0, 300))
    val testLabels = new CSVSequenceRecordReader
    testLabels.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath + "/test_label/"}%d.csv", 0, 300))

    val testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH)

    //Normalize the training data
    val normalizer = new NormalizerMinMaxScaler(0, 1)
    normalizer.fitLabel(true)
    normalizer.fit(trainData) //Collect training data statistics
    trainData.reset()

    println(testData.next())
    trainData.setPreProcessor(normalizer)
    testData.setPreProcessor(normalizer)

    val net = new model().get_model(regression)

    //Initialize the user interface backend
    val uiServer = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    net.setListeners(new StatsListener(statsStorage))

    net.addListeners(new ScoreIterationListener(100))

    val nEpochs = 100
    for (i <- 0 to nEpochs) {
      net.fit(trainData)
      //Evaluate on the test set:
//      val evaluation: RegressionEvaluation = net.evaluateRegression(testData)
//      println(s"======== EPOCH ${i} ========")
//      println(evaluation.stats())
      trainData.reset()
      testData.reset()

      if (i % 10 == 0) {

        var predicts: Array[Double] = Array()
        var actuals: Array[Double] = Array()

        while (trainData.hasNext) {
          val nextTestPoint = trainData.next
          val nextTestPointFeatures = nextTestPoint.getFeatures
          val predictionNextTestPoint = net.output(nextTestPointFeatures)

          val nextTestPointLabels = nextTestPoint.getLabels
          normalizer.revert(nextTestPoint)
          normalizer.revertLabels(predictionNextTestPoint)
                    println(
                      //s"Test point no.: ${nextTestPointFeatures} \n" +
                      s"Prediction is: ${predictionNextTestPoint} \n" +
                      s"Actual value is: ${nextTestPointLabels} \n")
          predicts = predicts :+ predictionNextTestPoint.getDouble(0L)
          actuals = actuals :+ nextTestPointLabels.getDouble(0L)
        }
        PlotUtil.plot(predicts, actuals, s"Train Run", i)

        predicts = Array()
        actuals = Array()
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
        net.save(new File(basePath.getAbsolutePath + "/model_mutil/" + i))
        testData.reset()
        trainData.reset()
      }
      net.rnnClearPreviousState()
    }

  }
}
