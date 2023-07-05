package predict

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator

import java.io.File
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize

object predict_OneDayAhead {
  def main(args: Array[String]): Unit = {
    val modelPath = new File("src/main/resources/data/model/")
    val forcastData = new File("src/main/resources/data/testData.csv")
    val net = MultiLayerNetwork.load(new File(modelPath.getAbsolutePath + "/" + "0"), true)

    val recordReader = new CSVRecordReader(0, ",")
    recordReader.initialize(new FileSplit(forcastData))

    val iterator = new RecordReaderDataSetIterator(recordReader, 1)

    val normalizer:NormalizerStandardize = new NormalizerStandardize()

    normalizer.load(
      new File(modelPath.getAbsolutePath + "/a"),
      new File(modelPath.getAbsolutePath + "/b"),
      new File(modelPath.getAbsolutePath + "/c"),
      new File(modelPath.getAbsolutePath + "/d")
    )

    iterator.setPreProcessor(normalizer)

    val Feature = iterator.next()

    val output = net.output(Feature.getFeatures)

    println(output)
  }
}
