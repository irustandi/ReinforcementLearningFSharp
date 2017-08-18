open System

let gamma = 0.99
let alpha = 0.01
let epsilon = 0.01
let initTheta = Array.create 7 1.0
Array.set initTheta 5 10.0

let states = [| 0 .. 6 |]
let isTerminal state = state >= 6

let rng = Random()

let getNextState state = 
    let nextState = if state < 5 then 5 
                    elif state = 5 then 
                        if rng.NextDouble() < epsilon then 6
                        else 5
                    else state
    (nextState, 0.0)

let getValue theta state =
    if isTerminal state then 0.0
    else
        let thetaCommon = Array.get theta 6
        if state < 5 then thetaCommon + 2.0 * (Array.get theta state)
        else 2.0 * thetaCommon + (Array.get theta state)

let deltaTemplate = Array2D.create 7 7 0.0
Array2D.set deltaTemplate 0 6 1.0
Array2D.set deltaTemplate 1 6 1.0
Array2D.set deltaTemplate 2 6 1.0
Array2D.set deltaTemplate 3 6 1.0
Array2D.set deltaTemplate 4 6 1.0
Array2D.set deltaTemplate 5 6 2.0
Array2D.set deltaTemplate 0 0 2.0
Array2D.set deltaTemplate 1 1 2.0
Array2D.set deltaTemplate 2 2 2.0
Array2D.set deltaTemplate 3 3 2.0
Array2D.set deltaTemplate 4 4 2.0
Array2D.set deltaTemplate 5 5 1.0

let update theta state nextState =
    let deltaThetaFun = Array2D.get deltaTemplate state
    let alphaError = alpha * (gamma * (getValue theta nextState) - (getValue theta state))
    let deltaTheta = Seq.map (fun i -> alphaError * (deltaThetaFun i)) (seq { for x=0 to 6 do yield x }) |> Seq.toArray
    Array.map2 (fun v1 v2 -> v1 + v2) theta deltaTheta