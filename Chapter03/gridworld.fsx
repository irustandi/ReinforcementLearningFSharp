open System

type Action = 
    | Left
    | Right
    | Up
    | Down

type State = { Top: int; Left: int }

type Size = { Width: int; Height: int }

let gamma = 0.9
let size = { Width = 5; Height = 5 }

let isOffGrid state = state.Left < 0 || state.Left >= size.Width || state.Top < 0 || state.Top >= size.Height

let basicMoveTo state action = 
    match action with
    | Up -> { Top = state.Top + 1; Left = state.Left }
    | Down -> { Top = state.Top - 1; Left = state.Left }
    | Left -> { Top = state.Top; Left = state.Left - 1 }
    | Right -> { Top = state.Top; Left = state.Left + 1 }

let isA state = state.Left = 1 && state.Top = 4
let isB state = state.Left = 3 && state.Top = 4

let moveToA = ({ Top = 0; Left = 1 }, 10.0)
let moveToB = ({ Top = 2; Left = 3 }, 5.0)

let moveTo state action =
    if isA state then moveToA
    elif isB state then moveToB
    else
        let nextState = basicMoveTo state action
        if isOffGrid nextState then (state, -1.0)
        else (nextState, 0.0)

