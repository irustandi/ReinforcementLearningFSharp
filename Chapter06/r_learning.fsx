open System

let n = 10
let h = 0.5
let p = 0.06

type Action = 
     | Accept
     | Reject

type State = { NumFreeServers: int; Priority: int}

type Experience = {
    State: State;
    Action: Action;
    Reward: float;
    NextState: State; }

let rng = Random()
let priorities = [|1; 2; 4; 8|]
let getNextPriority () = priorities.[rng.Next(priorities.Length)]

let rec getNextNumFreeServers numOccupiedServers numChangedServers = 
    if numOccupiedServers = 0 then numChangedServers
    else
        let nextNumChangedServers = 
            if rng.NextDouble() < p then numChangedServers + 1 else numChangedServers
        getNextNumFreeServers (numOccupiedServers - 1) nextNumChangedServers

let getNextState state action = 
    let numFreeServersNext = state.NumFreeServers + (getNextNumFreeServers (n - state.NumFreeServers) 0)
    let nextPriority = getNextPriority()
    match action with
    | Accept -> 
        if numFreeServersNext >= 1 then ({ NumFreeServers = min n (numFreeServersNext - 1); Priority = nextPriority }, float state.Priority)
        else ({ NumFreeServers = numFreeServersNext; Priority = nextPriority }, 0.0)
    | Reject -> ({ NumFreeServers = numFreeServersNext; Priority = nextPriority }, 0.0)

let alpha = 0.1
let beta = 0.01
let gamma = 1.0
let epsilon = 0.1

let choices = [| Accept; Reject |]
let randomDecide() = choices.[rng.Next(choices.Length)]
type Strategy = { State:State; Action:Action }
type QFunction = Map<Strategy, float>
type Parameters = { QFunction:QFunction; Rho:float }

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

let learn (parameters:Parameters) (exp:Experience) =
    let strat = { State = exp.State; Action = exp.Action }
    let nextAction = decide parameters.QFunction exp.NextState
    let valueAction = decideGreedy parameters.QFunction exp.NextState
    let qValueNext = qValue parameters.QFunction exp.NextState valueAction
    let qfuncNext = 
        match parameters.QFunction.TryFind strat with
        | Some(value) ->
            let value' = (1. - alpha) * value + alpha * (exp.Reward - parameters.Rho + gamma * qValueNext)
            parameters.QFunction.Add (strat, value')
        | None -> parameters.QFunction.Add (strat, alpha * (exp.Reward + gamma * qValueNext))
    let rhoNext = 
        if exp.Action = (decideGreedy qfuncNext exp.State)
        then (1. - beta) * parameters.Rho + beta * (exp.Reward + qValueNext - (qValue qfuncNext exp.State exp.Action))
        else parameters.Rho
    ({QFunction = qfuncNext; Rho = rhoNext}, nextAction)

let numTimes = 2000000

let rec loop state action parameters idx =
    if idx = numTimes then parameters
    else
        let (nextState, reward) = getNextState state action
        //let _ = Console.WriteLine("State: x:{0}, y:{1}; Action: {2}; Reward: {3}; NextState: x:{4}, y:{5}", state.Left, state.Top, getActionString action, reward, nextState.Left, nextState.Top)
        let (parametersNext, nextAction) = learn parameters {State=state; Action=action; Reward=reward; NextState = nextState}
        loop nextState nextAction parametersNext (idx+1)

let state = { NumFreeServers = n; Priority = getNextPriority() }
let action = Accept
let parametersFinal = loop state action {QFunction = Map.empty; Rho = 0.0} 0
