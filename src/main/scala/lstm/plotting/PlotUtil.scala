package lstm.plotting

import javax.swing._
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.chart.axis.{NumberAxis, NumberTickUnit}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}

object PlotUtil {
  def plot(predicts: Array[Double], actuals: Array[Double], name: String, epoch: Int): Unit = {

    val index = (0 until predicts.length).toList

    val min = (predicts union actuals).min
    val max = (predicts union actuals).max

    val dataSet = new XYSeriesCollection()

    addSeries(dataSet, index, predicts, "Predicts")
    addSeries(dataSet, index, actuals, "Actuals")

    val chart = ChartFactory.createXYLineChart(s"Prediction for training epoch ${epoch}", // chart title
      "Index", // x axis label
      name, // y axis label
      dataSet, // data
      PlotOrientation.VERTICAL,
      true, // include legend
      true, // tooltips
      false// urls
    )

    val xyPlot = chart.getXYPlot
    // X-axis
    val domainAxis = xyPlot.getDomainAxis.asInstanceOf[NumberAxis]
    domainAxis.setRange(index.head, index.reverse.head)
    domainAxis.setTickUnit(new NumberTickUnit(20))
    domainAxis.setVerticalTickLabels(true)

    // Y-axis
    val rangeAxis = xyPlot.getRangeAxis.asInstanceOf[NumberAxis]
    rangeAxis.setRange(min, max)
    rangeAxis.setTickUnit(new NumberTickUnit(50))
    val panel = new ChartPanel(chart)
    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setVisible(true)
  }

  private def addSeries(dataSet: XYSeriesCollection, x: List[Int], y: Array[Double], label: String): Unit = {
    val s = new XYSeries(label)
    x zip y map {
      case (x,y) => s.add(x,y)
    }
    dataSet.addSeries(s)
  }


}
