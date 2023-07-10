package predict

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import predict.train_model.{labelIndex, miniBatchSize, regression, testFiles, testSize}

import java.io.File
import train_model.test_visualization
import data_process.data_process

object test_model {
  val basePath = new File("src/main/resources/data/")
  val testFiles = new File(basePath, "test/")
  val modelPath = new File("src/main/resources/data/classify_model/")
  val forcastData = new File("src/main/resources/data/forcastData")

  val begin_index = 300
  val testSize = 40
  val miniBatchSize = 1
  val labelIndex = 0
  val regression = false
  def main(args: Array[String]): Unit = {
    val numPossibleLabels = if (regression) -1 else 2

    for(modelIndex <- 0 to 1000 by 10){
      println("testSize is " + modelIndex)

      new data_process().generate_test_data(regression)
      // Test Data
      val reader1 = new CSVSequenceRecordReader()
      reader1.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath}/%d.csv", begin_index, begin_index + testSize))

      val testData = new SequenceRecordReaderDataSetIterator(reader1, miniBatchSize, numPossibleLabels, labelIndex, regression)

      val net = MultiLayerNetwork.load(new File(modelPath.getAbsolutePath + "/" + modelIndex), true)

      val normalizer: NormalizerStandardize = new NormalizerStandardize()

      if (regression) {
        normalizer.fitLabel(true)
        normalizer.load(
          new File(modelPath.getAbsolutePath + "/a"),
          new File(modelPath.getAbsolutePath + "/b"),
          new File(modelPath.getAbsolutePath + "/c"),
          new File(modelPath.getAbsolutePath + "/d")
        )
        testData.setPreProcessor(normalizer)
      }

      test_visualization(testData, net, normalizer, 0)
    }
  }
}
