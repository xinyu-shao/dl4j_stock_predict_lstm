package predict

import java.io.File
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object predict_OneDayAhead {
  def main(args: Array[String]): Unit = {
    val modelPath = new File("src/main/resources/data/model/")

    val net = MultiLayerNetwork.load(new File(modelPath.getAbsolutePath + "/" + "0"), true)

    val Feature = null
  }
}
