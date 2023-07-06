package predict

import org.datavec.api.records.reader.impl.csv.{CSVRecordReader, CSVSequenceRecordReader}
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.deeplearning4j.datasets.datavec.{RecordReaderDataSetIterator, SequenceRecordReaderDataSetIterator}

import java.io.File
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize

object predict_OneDayAhead {
  def main(args: Array[String]): Unit = {
    val modelPath = new File("src/main/resources/data/model/")
    val forcastData = new File("src/main/resources/data/forcastData")
    val net = MultiLayerNetwork.load(new File(modelPath.getAbsolutePath + "/" + "200"), true)

    val miniBatchSize = 1
    val numPossibleLabels = -1
    val labelIndex = 0
    val regression = true

    // Training Data
    val reader = new CSVSequenceRecordReader()
    reader.initialize(new NumberedFileInputSplit(s"${forcastData}/%d.csv", 0, 1))

    val iterator = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, regression)

    val normalizer:NormalizerStandardize = new NormalizerStandardize()
    normalizer.fitLabel(true)
    normalizer.load(
      new File(modelPath.getAbsolutePath + "/a"),
      new File(modelPath.getAbsolutePath + "/b"),
      new File(modelPath.getAbsolutePath + "/c"),
      new File(modelPath.getAbsolutePath + "/d")
    )

    iterator.setPreProcessor(normalizer)

    val Data = iterator.next()
    val Feature = Data.getFeatures

    val output = net.output(Feature)

    normalizer.revert(Data)
    normalizer.revertLabels(output)

    println(Feature)
    println(Data.getLabels)
    println(output.getDouble(0L))
  }
}
