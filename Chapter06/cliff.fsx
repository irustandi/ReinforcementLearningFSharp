open System

type Action = 
    | Up
    | Down
    | Left 
    | Right

let getActionString action = 
    match action with
    | Up -> "Up"
    | Down -> "Down"
    | Left -> "Left"
    | Right -> "Right"

type State = { Top: int; Left: int }

type Size = { Width: int; Height: int }

type LearningAlgorithm = 
    | SARSA
    | QLearning

type Experience = {
    State: State;
    Action: Action;
    Reward: float;
    NextState: State; }

let size = { Width = 12; Height = 4 }
let initState = { Top = 0; Left = 0 }
let terminalState = { Top = 0; Left = 11 }

let rng = Random()

let isEqual state1 state2 = 
    state1.Top = state2.Top && state1.Left = state2.Left

let isCliff state = (not (isEqual state initState)) && (not (isEqual state terminalState)) && state.Top = 0

let rewardInfo = Array2D.create size.Width size.Height -1.0
[ for y in 1 .. 10 -> Array2D.set rewardInfo y 0 -100.0 ]

let boundState state = 
    { Top = min (size.Height - 1) (max 0 state.Top); Left = min (size.Width - 1) (max 0 state.Left) }

let basicMoveTo state action = 
    match action with
    | Up -> { Top = state.Top + 1; Left = state.Left }
    | Down -> { Top = state.Top - 1; Left = state.Left }
    | Left -> { Top = state.Top; Left = state.Left - 1 }
    | Right -> { Top = state.Top; Left = state.Left + 1 }

let adjustState state = if isCliff state then initState else state

let moveTo state action =
    let nextState = basicMoveTo state action |> boundState
    let reward = Array2D.get rewardInfo nextState.Left nextState.Top
    if isCliff nextState then (initState, reward) else (nextState, reward)

type Strategy = { State:State; Action:Action }
type QFunction = Map<Strategy, float>

let choices = [| Up; Down; Left; Right |]
let randomDecide() = choices.[rng.Next(choices.Length)]
let epsilon = 0.1
let alpha = 0.1
let gamma = 1.0

let qValue (qfunc:QFunction) state action = 
    match qfunc.TryFind {State=state; Action=action} with
    | Some (value) -> value
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

let learningAlgorithm = QLearning
//let learningAlgorithm = SARSA
let learn (qfunc:QFunction) (exp:Experience) =
    let strat = { State = exp.State; Action = exp.Action }
    let nextAction = decide qfunc exp.NextState
    let valueAction = 
        match learningAlgorithm with
        | SARSA -> nextAction
        | QLearning -> decideGreedy qfunc exp.NextState
    let qValueNext = if isEqual exp.NextState terminalState then 0.0 else qValue qfunc exp.NextState valueAction
    let qfuncNext = 
        match qfunc.TryFind strat with
        | Some(value) ->
            let value' = (1. - alpha) * value + alpha * (exp.Reward + gamma * qValueNext)
            qfunc.Add (strat, value')
        | None -> qfunc.Add (strat, alpha * (exp.Reward + gamma * qValueNext))
    (qfuncNext, nextAction)

let numEpisodes = 5000

let rec simulate qfunc episodeIdx = 
    if episodeIdx = numEpisodes then qfunc
    else
        let state = initState
        let action = decide qfunc state

        let rec loop state action qfunc episodeLen =
            let (nextState, reward) = moveTo state action
            //let _ = Console.WriteLine("State: x:{0}, y:{1}; Action: {2}; Reward: {3}; NextState: x:{4}, y:{5}", state.Left, state.Top, getActionString action, reward, nextState.Left, nextState.Top)
            let (qfuncNext, nextAction) = learn qfunc {State=state; Action=action; Reward=reward; NextState = nextState}
            if isEqual nextState terminalState then (qfuncNext, episodeLen)
            else loop nextState nextAction qfuncNext (episodeLen+1)

        let (qfuncNext, episodeLen) = loop state action qfunc 1
        //let _ = Console.WriteLine("episode: {0}, length: {1}", episodeIdx, episodeLen)
        simulate qfuncNext (episodeIdx+1)

let rec doGreedy qfunc state =
    let action = decideGreedy qfunc state
    let value = qValue qfunc state action
    let (nextState, _) = moveTo state action
    let _ = Console.WriteLine("State: (x {0}, y {1}), Action: {2}, Value: {3}, Next State: (x {4}, y {5})", state.Left, state.Top, getActionString action, value, nextState.Left, nextState.Top)
    if isEqual nextState terminalState then () else doGreedy qfunc nextState

doGreedy (simulate Map.empty 0) initState