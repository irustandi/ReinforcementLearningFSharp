open System

type Action =
    | Up
    | Down
    | Left
    | Right

type Size = { Width: int; Height: int }
type State = { Left: int; Top: int }

type Experience = {
    State: State;
    Action: Action;
    Reward: float;
    NextState: State; }

let size = { Width = 9; Height = 6 }
let initState = { Left = 0; Top = 3 }
let terminalState = { Left = 8; Top = 5}

let rewardInfo = Array2D.create size.Width size.Height 0.0
Array2D.set rewardInfo terminalState.Left terminalState.Top 1.0

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

let wallMap = Map.empty
              |> Map.add { Left = 2; Top = 2 } true
              |> Map.add { Left = 2; Top = 3 } true
              |> Map.add { Left = 2; Top = 4 } true
              |> Map.add { Left = 5; Top = 1 } true
              |> Map.add { Left = 7; Top = 3 } true
              |> Map.add { Left = 7; Top = 4 } true
              |> Map.add { Left = 7; Top = 5 } true

let isWall state =
    match wallMap.TryFind state with
    | Some(value) -> value
    | None -> false

let moveTo state action =
    let nextState = basicMoveTo state action |> boundState |> (fun st -> if isWall st then state else st)
    let reward = Array2D.get rewardInfo nextState.Left nextState.Top
    (nextState, reward)

type Strategy = { State: State; Action: Action }
type QFunction = Map<Strategy, float>
type ModelFunction = Map<Strategy, State * float>
type StateSet = Set<State>
type StateActionMap = Map<State, Set<Action>>
type TimeFunction = Map<Strategy, int>

type LearningAlgorithm = DynaQ of QFunction * ModelFunction * StateSet * StateActionMap
                       | DynaQPlus of LearningAlgorithm * int * TimeFunction

let N = 30
let alpha = 0.1
let epsilon = 0.1
let gamma = 0.95
let rng = Random()

let qValue (qfunc:QFunction) state action =
    match qfunc.TryFind {State=state; Action=action} with
    | Some (value) -> value
    | None -> 0.

let choices = [| Up; Down; Left; Right |]

let randomDecide() = choices.[rng.Next(choices.Length)]

let decideGreedy algo state =
    match algo with
    | DynaQ(qfunc, modelfunc, _, _) ->
        let eval =
            choices
            |> Array.map (fun alt -> { State = state; Action = alt })
            |> Array.filter (fun strat -> qfunc.ContainsKey strat)
        match eval.Length with
            | 0 -> randomDecide()
            | _ ->
                choices
                |> Seq.maxBy (fun alt -> qValue qfunc state alt)

let updateAlgo algo (exp:Experience) nextAction =
    let strat = { State=exp.State; Action=exp.Action }
    match algo with
    | DynaQ(qfunc, modelfunc, stateSet, stateActionMap) ->
        let qValueNext = qValue qfunc exp.NextState nextAction
        let qfuncNext =
            match qfunc.TryFind strat with
            | Some(value) ->
                let value' = (1. - alpha) * value + alpha * (exp.Reward + gamma * qValueNext)
                qfunc.Add (strat, value')
            | None -> qfunc.Add (strat, alpha * (exp.Reward + gamma * qValueNext))
        let modelfuncNext = modelfunc.Add (strat, (exp.NextState, exp.Reward))
        let stateSetNext = Set.add exp.State stateSet
        let actionSet = match Map.tryFind exp.State stateActionMap with
                        | Some(value) -> value
                        | None -> Set.empty
        let stateActionMapNext = Map.add exp.State (Set.add exp.Action actionSet) stateActionMap
        DynaQ(qfuncNext, modelfuncNext, stateSetNext, stateActionMapNext)

let sampleState algo =
    match algo with
    | DynaQ(_, _, stateSet, _) ->
        let states = stateSet |> Set.toArray
        states.[rng.Next(states.Length)]

let sampleActionForState algo state =
    match algo with
    | DynaQ(_, _, _, stateActionMap) ->
        let actionArray = match Map.tryFind state stateActionMap with
                              | Some(value) -> value |> Set.toArray
                              | None -> Array.empty
        actionArray.[rng.Next(actionArray.Length)]

let getModelValue (modelfunc:ModelFunction) strategy = Map.find strategy modelfunc

let randomSample algo =
    let state = sampleState algo
    let action = sampleActionForState algo state
    let strat = {State = state; Action = action}
    match algo with
    | DynaQ(qfunc, modelfunc, stateSet, stateActionMap) ->
        let (nextState, reward) = getModelValue modelfunc strat
        let nextAction = decideGreedy algo nextState
        let qValueNext = qValue qfunc nextState nextAction
        let qfuncNext =
            match qfunc.TryFind strat with
            | Some(value) ->
                let value' = (1. - alpha) * value + alpha * (reward + gamma * qValueNext)
                qfunc.Add (strat, value')
            | None -> qfunc.Add (strat, alpha * (reward + gamma * qValueNext))
        DynaQ(qfuncNext, modelfunc, stateSet, stateActionMap)

let learn algo (exp:Experience) =
    let nextAction = decideGreedy algo exp.NextState
    let algoNext = updateAlgo algo exp nextAction
    Seq.fold (fun alg _ -> randomSample alg) algoNext (seq { for n = 1 to N do yield n })
