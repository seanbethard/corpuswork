import scala.io.Source

object EntryPoint {

  def main(args: Array[String]) {
    val lexicon = new Lexicon
    println("Vocab size: " + lexicon.vocab.size)
    val grammar = new Grammar("example/CFG.txt")
    var tokens = Source.fromFile("example/sentence.txt").getLines().next().split(" ")
    val table = new WFST(grammar, lexicon, tokens).displayTable
  }
}