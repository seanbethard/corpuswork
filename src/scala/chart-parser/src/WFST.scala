// A simple bottom-up recognizer that determines whether an input string is in the language defined by a grammar. The algorithm proceeds by building up constituents from an input string based on production rules in a grammar. Intermediate results, namely those constituents that have been built so far, are kept in a well-formed substring table. An "S" in the top-right cell of the table indicates that a parse exists over the whole input. All grammar productions must be binary. Lexical items are grouped into unique lexical categories.

// For help with the implementation I referred to the following sources.

// Chapter 8, Natural Language Processing with Python
// Chapter 11, Techniques in Natural Language Processing I
// Natural Language Processing Techniques in Prolog


import java.util.NoSuchElementException

class WFST(grammar: Grammar, lexicon: Lexicon, tokens: Array[String]) {

  val table = Array.fill[String](tokens.length, tokens.length) {null}

  for (i <- 0 to table.length - 1) {
    try {
      val tmp = lexicon.vocab.get(tokens(i)).get.iterator
      table(i)(i) = tmp.next()
    }
    catch {
      case oov: NoSuchElementException =>
        println("Term \""+tokens(i)+"\" not found.")
    }
  }

  for (j <- Range(2, tokens.length + 1)) {
    for (i <- Range(0, tokens.length + 1 - j)) {
      val end = i + j
      for (k <- Range(i + 1, end)) {
        val nt1: String = table(i)(k-1)
        val nt2: String = table(k)(end-1)
        if (grammar.productions.contains(List(nt1, nt2))) {
          val lhs = grammar.productions.get(List(nt1, nt2)).get
          table(i)(end - 1) = lhs
          printTrace( lhs, nt2, nt1,
            k.toString, i.toString, end.toString)
        }
      }
    }
  }

  def displayTable {
    print("\nWFST ")
    for (i: Int <- Range(1,table.length+1)) {
      if (i >= 10)
        print(i + " " * 3)
      else
        print(i + " " * 4)
    }
    println()
    var i = 0
    table foreach { row =>
      println()
      print(i)
      if(i >= 10)
        print(" " * 3)
      else
        print(" " * 4)
      row foreach { item =>
        if (item == null) {
          print("*" + " " * 4)
        } else if (item.length == 2) {
          print(item + " " * (item.length + 1))
        } else if (item.length > 2) {
          print(item + " " * (item.length - 1))
        } else print(item + " " * 4)
      }
      i += 1
      println()
    }
    println()
  }

  def printTrace(x: String,
                 y: String,
                 z: String,
                 a: String,
                 b: String,
                 c: String) {
    println(b+" "+z+" "+a+" "+y+" "+c+" -> "+b+" "+x+" "+c)
  }
}