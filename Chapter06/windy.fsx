open System
    
type Action = 
    | Up
    | Down
    | Left 
    | Right
    //| LeftUp
    //| LeftDown
    //| RightUp
    //| RightDown
    //| Stay

type State = { Top: int; Left: int }

type Size = { Width: int; Height: int }

type Experience = {
    State: State;
    Action: Action;
    Reward: float;
    NextState: State; }

let size = { Width = 10; Height = 7 }
let initState = { Top = 3; Left = 0 }
let terminalState = { Top = 3; Left = 7 }
let rewardInfo = Array2D.create size.Width size.Height -1.0
Array2D.set rewardInfo terminalState.Left terminalState.Top 1.0

let windInfo = Array.create size.Width 0
Array.set windInfo 3 1
Array.set windInfo 4 1
Array.set windInfo 5 1
Array.set windInfo 6 2
Array.set windInfo 7 2
Array.set windInfo 8 1

let rng = Random()
let isStochastic = true

let isTerminal state = 
    state.Top = terminalState.Top && state.Left = terminalState.Left

let boundState state = 
    { Top = min (size.Height - 1) (max 0 state.Top); Left = min (size.Width - 1) (max 0 state.Left) }

let basicMoveTo state action = 
    match action with
    | Up -> { Top = state.Top + 1; Left = state.Left }
    | Down -> { Top = state.Top - 1; Left = state.Left }
    | Left -> { Top = state.Top; Left = state.Left - 1 }
    | Right -> { Top = state.Top; Left = state.Left + 1 }
    //| LeftUp -> { Top = state.Top + 1; Left = state.Left - 1 }
    //| LeftDown -> { Top = state.Top - 1; Left = state.Left - 1 }
    //| RightUp -> { Top = state.Top + 1; Left = state.Left + 1 }
    //| RightDown -> { Top = state.Top - 1; Left = state.Left + 1 }
    //| Stay -> state

let getWind xCoord isStochastic = 
    let rawWind = Array.get windInfo xCoord
    if isStochastic then rawWind + (rng.Next(3) - 1) else rawWind

let moveTo state action =
    let rawNextState = basicMoveTo state action |> boundState
    let windNextState = { Top = rawNextState.Top + (getWind rawNextState.Left isStochastic); Left = rawNextState.Left } |> boundState
    let reward = Array2D.get rewardInfo windNextState.Left windNextState.Top
    (windNextState, reward)

type Strategy = { State:State; Action:Action }
type QFunction = Map<Strategy, float>
type EligibilityTrace = Map<Strategy, float>

type LearningAlgorithm = 
    | SARSA of QFunction
    | SARSALambda of QFunction * float * EligibilityTrace

let createSARSA () = SARSA(Map.empty)

let createSARSALambda lambda = SARSALambda(Map.empty, lambda, Map.empty)

let getQFunction algorithm = 
    match algorithm with
    | SARSA(qfunc) -> qfunc
    | SARSALambda(qfunc, _, _) -> qfunc
let choices = [| Up; Down; Left; Right |]
//let choices = [| Up; Down; Left; Right; LeftUp; LeftDown; RightUp; RightDown; Stay |]
let randomDecide() = choices.[rng.Next(choices.Length)]
let epsilon = 0.1
let alpha = 0.1
let gamma = 1.0

let qValue (qfunc:QFunction) state action = 
    match qfunc.TryFind {State=state; Action=action} with
    | Some (value) -> value
    | None -> 0.

let eValue (es:EligibilityTrace) state action = 
    match es.TryFind {State=state; Action=action} with
    | Some(value) -> value
    | None -> 0.

let decideGreedy (qfunc:QFunction) (state:State) = 
    let eval = 
        choices
        |> Array.map (fun alt -> { State = state; Action = alt })
        |> Array.filter (fun strat -> qfunc.ContainsKey strat)
    match eval.Length with
    | 0 -> randomDecide()
    | _ ->
        choices
        |> Seq.maxBy (fun alt -> qValue qfunc state alt)

let decide (qfunc:QFunction) (state:State) = 
    if (rng.NextDouble() < epsilon) then randomDecide()
    else decideGreedy qfunc state

let allStates = seq { for x in 1 .. size.Width do
                        for y in 1 .. size.Height do
                            yield {Top = y-1; Left = x-1} }
let allStateActions = seq { for state in allStates do
                                for action in choices do
                                    yield {State = state; Action = action} }

let addQFunction (qfunc:QFunction) strat addedValue =
    match qfunc.TryFind strat with  
    | Some(value) ->
        let value' = value + addedValue
        qfunc.Add(strat, value')
    | None -> qfunc.Add(strat, addedValue)

// SARSA
let learn algo (exp:Experience) =
    let strat = { State = exp.State; Action = exp.Action }
    let qfunc = getQFunction algo
    let nextAction = decide qfunc exp.NextState
    let qValueNext = if isTerminal exp.NextState then 0.0 else qValue qfunc exp.NextState nextAction
    match algo with
    | SARSA(_) ->
        let qfuncNext = 
            match qfunc.TryFind strat with
            | Some(value) ->
                let value' = (1. - alpha) * value + alpha * (exp.Reward + gamma * qValueNext)
                qfunc.Add (strat, value')
            | None -> qfunc.Add (strat, alpha * (exp.Reward + gamma * qValueNext))
        (SARSA(qfuncNext), nextAction)
    | SARSALambda(_, lambda, es) ->
        let delta = exp.Reward + gamma * qValueNext - (qValue qfunc exp.State exp.Action)
        let _ = es.Add (strat, 1.0 + (eValue es strat.State strat.Action))
        let (qfFinal, esFinal) = Seq.fold (fun (qfCurr, esCurr) str -> 
                                               let eVal = eValue esCurr str.State str.Action
                                               let qfNext = addQFunction qfCurr str (alpha * lambda * eVal)
                                               let esNext = esCurr.Add (str, gamma * lambda * eVal)
                                               (qfNext, esNext)) (qfunc, es) allStateActions
        (SARSALambda(qfFinal, lambda, esFinal), nextAction)

let numEpisodes = 5000

let rec simulate algo episodeIdx = 
    if episodeIdx = numEpisodes then algo
    else
        let qfunc = getQFunction algo
        let state = initState
        let action = decide qfunc state

        let rec loop state action algo episodeLen =
            let (nextState, reward) = moveTo state action
            //let _ = printfn "State: x:%d, y:%d; Action: %A; Reward: %g; NextState: x:%d, y:%d", state.Left, state.Top, action, reward, nextState.Left, nextState.Top
            let (qfuncNext, nextAction) = learn algo {State=state; Action=action; Reward=reward; NextState = nextState}
            if isTerminal nextState then (qfuncNext, episodeLen)
            else loop nextState nextAction qfuncNext (episodeLen+1)

        let (qfuncNext, episodeLen) = loop state action algo 1
        //let _ = printfn "episode: {0}, length: {1}", episodeIdx, episodeLen
        simulate qfuncNext (episodeIdx+1)

let rec doGreedy qfunc state =
    let action = decideGreedy qfunc state
    let value = qValue qfunc state action
    let (nextState, _) = moveTo state action
    let _ = printfn "State: (x %d, y %d), Action: %A, Value: %g, Next State: (x %d, y %d)", state.Left, state.Top, action, value, nextState.Left, nextState.Top
    if isTerminal nextState then () else doGreedy qfunc nextState

let qfuncFinal = simulate (Map.empty) 0
qfuncFinal.TryFind({State = initState; Action=Up})
qfuncFinal.TryFind({State = initState; Action=Down})
qfuncFinal.TryFind({State = initState; Action=Left})
qfuncFinal.TryFind({State = initState; Action=Right})

qfuncFinal.TryFind({State = {Top = 3; Left = 1}; Action=Up})
qfuncFinal.TryFind({State = {Top = 3; Left = 1}; Action=Down})
qfuncFinal.TryFind({State = {Top = 3; Left = 1}; Action=Left})
qfuncFinal.TryFind({State = {Top = 3; Left = 1}; Action=Right})

qfuncFinal.TryFind({State = {Top = 1; Left = 2}; Action=Left})
qfuncFinal.TryFind({State = {Top = 1; Left = 2}; Action=Right})
qfuncFinal.TryFind({State = {Top = 1; Left = 2}; Action=Up})
qfuncFinal.TryFind({State = {Top = 1; Left = 2}; Action=Down})

doGreedy (simulate Map.empty 0) initState