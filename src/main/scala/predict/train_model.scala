package predict

import lstm.plotting.PlotUtil
import org.datavec.api.records.reader.impl.csv._
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.dataset.api.preprocessor.{NormalizerMinMaxScaler, NormalizerStandardize}
import data_process.data_process
import model.model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import java.io.File

object train_model {
  val basePath = new File("src/main/resources/data/")
  val trainingFiles = new File(basePath, "train/")
  val testFiles = new File(basePath, "test/")

  val regression = false
  val trainSize = 1200
  val testSize = 500
  val miniBatchSize = 1
  val labelIndex = 0
  def main(args:Array[String]): Unit ={

    val numPossibleLabels = if(regression) -1 else 2

    new data_process().generate_data(regression)

    // Training Data
    val reader = new CSVSequenceRecordReader()
    reader.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath}/%d.csv", 0, trainSize))

    val trainData = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, regression)


    // Test Data
    val reader1 = new CSVSequenceRecordReader()
    reader1.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath}/%d.csv", 0, testSize))

    val testData = new SequenceRecordReaderDataSetIterator(reader1, miniBatchSize, numPossibleLabels, labelIndex, regression)

    //Normalize the training data

    val normalizer = new NormalizerStandardize()
    if (regression) {
      normalizer.fitLabel(regression)
      normalizer.fit(trainData) //Collect training data statistics
      trainData.reset()

      normalizer.save(
        new File(basePath.getAbsolutePath + "/regression_model/a"),
        new File(basePath.getAbsolutePath + "/regression_model/b"),
        new File(basePath.getAbsolutePath + "/regression_model/c"),
        new File(basePath.getAbsolutePath + "/regression_model/d")
      )

      trainData.setPreProcessor(normalizer)
      testData.setPreProcessor(normalizer)
    }

    val net = new model().get_model(regression)

    val uiServer = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    net.setListeners(new StatsListener(statsStorage))

    if(regression)net.addListeners(new ScoreIterationListener(100))

    val nEpochs = 1000
    val judge = if(regression) 100 else 10
    for (i <- 0 to nEpochs) {
      net.fit(trainData)
      trainData.reset()
      //Evaluate on the test set:

      println(s"======== EPOCH ${i} ========")
      if(regression){
        val evaluation: RegressionEvaluation = net.evaluateRegression(testData)
        println(evaluation.stats())
        testData.reset()
      }
      net.rnnClearPreviousState()
      if (i % judge == 0) {
        test_visualization(trainData, net, normalizer, i)
        test_visualization(testData, net, normalizer, i)
        net.rnnClearPreviousState()
      }
    }
  }
  def test_visualization(Data:SequenceRecordReaderDataSetIterator, net: MultiLayerNetwork, normalizer:NormalizerStandardize, Epoch: Int)={
    var predicts: Array[Double] = Array()
    var actuals: Array[Double] = Array()

    while (Data.hasNext) {
      val nextTestPoint = Data.next
      val nextTestPointFeatures = nextTestPoint.getFeatures
      val predictionNextTestPoint = net.output(nextTestPointFeatures, false)

      val nextTestPointLabels = nextTestPoint.getLabels
      if (regression) {
        normalizer.revert(nextTestPoint) // revert the normalization of this test point
        normalizer.revertLabels(predictionNextTestPoint)
        for(i <- 0L to predictionNextTestPoint.length() - 1){
          predicts = predicts :+ predictionNextTestPoint.getDouble(i)
          actuals = actuals :+ nextTestPointLabels.getDouble(i)
        }
      } else {
        for (i <- 1L to predictionNextTestPoint.length()) {
          if(i %2 != 0){
            val predict = if (predictionNextTestPoint.getDouble(i - 1) > 0.5) 1.0 else 0.0
            predicts = predicts :+ predict
            actuals = actuals :+ nextTestPointLabels.getDouble(i - 1)
          }
        }
      }
    }
    if (regression) {
      PlotUtil.plot(predicts, actuals, s"Test Run", Epoch)
    } else {
      var right = 0
      for ((a, b) <- predicts zip actuals) {
        if (a == b) {
          right += 1
        }
      }
      println("精准度为：" + right.toDouble / predicts.length)
    }
    Data.reset()
  }
}
