open System
#load "TileCoding.fs"
open TileCoding

// Game parameters and functions
let minPosition = -1.2
let maxPosition = 0.6
let minVelocity = -0.07
let maxVelocity = 0.07
let goalPosition = 0.5

type State = { position: float; velocity: float }

type Action = 
    | FullThrottleForward
    | FullThrottleReverse
    | ZeroThrottle

type Experience = {
    state: State;
    action: Action;
    reward: float;
    nextState: State; }

let getActionValue action = 
    match action with
    | FullThrottleForward -> 1.0
    | FullThrottleReverse -> -1.0
    | ZeroThrottle -> 0.0

let isAtGoal state = state.position >= goalPosition

let boundVariable maxValue minValue value = 
    if value > maxValue then maxValue
    elif value < minValue then minValue
    else value

let takeAction state action =
    let actionVal = getActionValue action
    let newVelocity = state.velocity + (actionVal-1.0) * 0.001 - 0.0025 * (Math.Cos (3.0 * state.position)) 
                      |> boundVariable maxVelocity minVelocity
    let newPosition = state.position + newVelocity  
                      |> boundVariable maxPosition minPosition
    let newState = if newPosition <= minPosition && newVelocity < 0.0 then { position=newPosition; velocity=0.0 }
                   else { position=newPosition; velocity=newVelocity }
    let reward = if isAtGoal newState then 1.0 else -1.0
    (newState, reward)

let initState = { position = -0.5; velocity = 0.0 }

// Strategy parameters and functions
let epsilon = 0.0 // probability of random actions
let gamma = 1.0 // discount rate
let alpha = 0.5 // step-size parameter
let lambda = 0.9 // trace-decay parameter

let numTilings = 10
let numPos = 8
let numVel = 8
let posWidth = (goalPosition - minPosition) / (float numPos)
let velWidth = (maxVelocity - minVelocity) / (float numVel)

let choices = [| FullThrottleForward; FullThrottleReverse; ZeroThrottle |]
let numFeatures = numTilings * (numPos + 2) * (numVel + 2)
let numActions = Array.length choices
let memorySize = numActions * numFeatures
let maxNumTimesteps = 1000

let rng = Random()

let useAccTrace = true

type FMap = Map<Action, int []>
type QMap = Map<Action, float>

let loadFMap state =
    let stateArray = Array.create 2 0.0
    let _ = Array.set stateArray 0 (state.position / posWidth)
    let _ = Array.set stateArray 1 (state.velocity / velWidth)

    Array.fold (fun (f:FMap) a -> 
                    let tiles = Tiles.getTiles numTilings stateArray memorySize
                    f.Add(a, tiles)) Map.empty choices

let getTilesFromFMap (fMap:FMap) action = 
    match fMap.TryFind action with
    | Some(value) -> value
    | None -> Array.create 0 0

let loadQMap theta (fMap:FMap) = 
    choices
    |> Array.fold (fun (q:QMap) a ->
                      let qVal = getTilesFromFMap fMap a |> Array.fold (fun s m -> s + (Array.get theta m)) 0.0
                      q.Add(a, qVal)) Map.empty

let getQValueFromQMap action (qMap:QMap) = 
    match qMap.TryFind action with
    | Some(value) -> value
    | None -> 0.0

type TileCodingApproximationWithEligibilityTrace = { theta: float []; es:float [] }

let randomDecide() = choices.[rng.Next(choices.Length)]

let decideGreedy algo (state:State) =
    let qMap = loadFMap state |> loadQMap algo.theta
    choices
    |> Seq.maxBy (fun action -> getQValueFromQMap action qMap)

let decide algo (state:State) = 
    if (rng.NextDouble() < epsilon) then randomDecide()
    else decideGreedy algo state

let learn algo (exp:Experience) =
    let fMap = loadFMap exp.state
    let qValue = fMap |> loadQMap algo.theta |> getQValueFromQMap exp.action
    let delta = exp.reward - qValue
    let nextAction = decide algo exp.nextState
    let esNext = algo.es |> Array.map (fun v -> v * gamma * lambda)
    let _ = choices 
            |> Array.map (fun action -> if action = exp.action then ()
                                        else 
                                            let _ = getTilesFromFMap fMap action
                                                    |> Array.map (fun tileValue -> Array.set esNext tileValue 0.0)
                                            ())
    let _ = getTilesFromFMap fMap exp.action
            |> Array.map (fun tileValue -> if useAccTrace then Array.set esNext tileValue (1.0 + (Array.get esNext tileValue))
                                           else Array.set esNext tileValue 1.0)    
    let thetaNext = algo.theta
                    |> Array.mapi (fun idx value -> value + (alpha/(float numTilings)) * delta * (Array.get esNext idx))
    ({ theta = thetaNext; es = esNext }, nextAction)

let createAlgorithm () = { theta = Array.create memorySize 0.0; es = Array.create memorySize 0.0 }

let numEpisodes = 10

let rec simulate algo episodeIdx = 
    if episodeIdx = numEpisodes then algo
    else
        let state = initState
        let action = decide algo state

        let rec loop state action algo episodeLen =
            let (nextState, reward) = takeAction state action
            //let _ = Console.WriteLine("State: x:{0}, y:{1}; Action: {2}; Reward: {3}; NextState: x:{4}, y:{5}", state.Left, state.Top, getActionString action, reward, nextState.Left, nextState.Top)
            let (algoNext, nextAction) = learn algo {state=state; action=action; reward=reward; nextState = nextState}
            if isAtGoal nextState || episodeLen = maxNumTimesteps then (algoNext, episodeLen)
            else loop nextState nextAction algoNext (episodeLen+1)

        let (algoNext, episodeLen) = loop state action algo 1
        //let _ = Console.WriteLine("episode: {0}, length: {1}", episodeIdx, episodeLen)
        simulate algoNext (episodeIdx+1)

