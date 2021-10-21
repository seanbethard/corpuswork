module Sentiment exposing (..)

-- Input a sentence for sentiment analysis.

import Browser
import Html exposing (Html, Attribute, div, input, text)
import Html.Attributes exposing (..)
import Html.Events exposing (onInput)


-- MAIN


main =
  Browser.sandbox { init = init, update = update, view = view }



-- MODEL


type alias Model =
  { sentence : String
  }


init : Model
init =
  { sentence = "" }



-- UPDATE


type Msg
  = Name String


update : Msg -> Model -> Model
update msg model =
  case msg of
    Name sentence ->
      { model | sentence = sentence }


-- VIEW


view : Model -> Html Msg
view model =
  div []
    [ viewSentence "sentence" "Sentence" model.sentence Name
    , div [] [ text (model.sentence) ]
    , viewSentenceValidation model
    ]

viewSentence : String -> String -> String -> (String -> msg) -> Html msg
viewSentence t p v toMsg =
  input [ type_ t, placeholder p, value v, onInput toMsg ] []

viewSentenceValidation : Model -> Html msg
viewSentenceValidation model =
  if isTooLong model.sentence then
    div [ style "color" "red"] [ text "Sentence is too long!"]
  else
    div [ style "color" "green" ] [ text "Sentence not too long."]


isTooLong : String -> Bool
isTooLong sentence =
  if String.length sentence > 128 then
    True
  else
    False

