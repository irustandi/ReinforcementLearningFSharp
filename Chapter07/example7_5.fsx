open System

type Action = 
    | Right
    | Wrong

let getActionString action = 
    match action with
    | Right -> "Right"
    | Wrong -> "Wrong"

type State = int

type Experience = {
    State: State;
    Action: Action;
    Reward: float;
    NextState: State; }

let initState:State = 0
let terminalState:State = 5
let isTerminal (state:State) = (terminalState = state)
let takeAction (state:State) action =
    let nextState = 
        match action with
        | Right -> state + 1
        | Wrong -> state
    let reward = if isTerminal nextState then 1.0 else 0.0
    (nextState, reward)

type Strategy = { State:State; Action:Action }
type QFunction = Map<Strategy, float>
type EligibilityTrace = Map<Strategy, float>

type LearningAlgorithm = 
    | SARSALambda of QFunction * float * float * bool * EligibilityTrace

let createSARSALambda alpha lambda useAccTrace = SARSALambda(Map.empty, alpha, lambda, useAccTrace, Map.empty)

let getQFunction algorithm = 
    match algorithm with
    | SARSALambda(qfunc, _, _, _, _) -> qfunc

let rng = Random()
let choices = [| Right; Wrong |]
//let choices = [| Up; Down; Left; Right; LeftUp; LeftDown; RightUp; RightDown; Stay |]
let randomDecide() = choices.[rng.Next(choices.Length)]
let epsilon = 0.1
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

let allStates = seq { for x in 0 .. (terminalState-1) do yield x }
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
    | SARSALambda(_, alpha, lambda, useAccTrace, es) ->
        let delta = exp.Reward + gamma * qValueNext - (qValue qfunc exp.State exp.Action)
        let _ = if useAccTrace then es.Add (strat, 1.0 + (eValue es strat.State strat.Action))
                else es.Add (strat, 1.0)
        let (qfFinal, esFinal) = Seq.fold (fun (qfCurr, esCurr) str -> 
                                               let eVal = eValue esCurr str.State str.Action
                                               let qfNext = addQFunction qfCurr str (alpha * lambda * eVal)
                                               let esNext = esCurr.Add (str, gamma * lambda * eVal)
                                               (qfNext, esNext)) (qfunc, es) allStateActions
        (SARSALambda(qfFinal, alpha, lambda, useAccTrace, esFinal), nextAction)

let numEpisodes = 5000

let rec simulate algo episodeIdx = 
    if episodeIdx = numEpisodes then algo
    else
        let qfunc = getQFunction algo
        let state = initState
        let action = decide qfunc state

        let rec loop state action algo episodeLen =
            let (nextState, reward) = takeAction state action
            //let _ = Console.WriteLine("State: x:{0}, y:{1}; Action: {2}; Reward: {3}; NextState: x:{4}, y:{5}", state.Left, state.Top, getActionString action, reward, nextState.Left, nextState.Top)
            let (qfuncNext, nextAction) = learn algo {State=state; Action=action; Reward=reward; NextState = nextState}
            if isTerminal nextState then (qfuncNext, episodeLen)
            else loop nextState nextAction qfuncNext (episodeLen+1)

        let (qfuncNext, episodeLen) = loop state action algo 1
        //let _ = Console.WriteLine("episode: {0}, length: {1}", episodeIdx, episodeLen)
        simulate qfuncNext (episodeIdx+1)

let rec doGreedy qfunc state =
    let action = decideGreedy qfunc state
    let value = qValue qfunc state action
    let (nextState, _) = takeAction state action
    let _ = Console.WriteLine("State: (x {0}, y {1}), Action: {2}, Value: {3}, Next State: (x {4}, y {5})", state.Left, state.Top, getActionString action, value, nextState.Left, nextState.Top)
    if isTerminal nextState then () else doGreedy qfunc nextState
