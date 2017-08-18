#load "../packages/FsLab/FsLab.fsx"

open System
open Deedle
open XPlot.GoogleCharts
open XPlot.GoogleCharts.Deedle

type Strategy =
    | TD of float * float [] * int
    | TDNStepsOnline of float * float [] * int * int list * float list
    | TDNStepsOffline of float * float [] * int * int list * float list * float []
    | LambdaReturn of float * float * bool * float [] * int list * float list
    | TDLambda of float * float * bool * float [] * float [] * int
    | ConstantAlphaMC of float * float [] * int list * float
    | Batch of Strategy * int list list * float list list

let rnd = Random()

let numStates = 19

let getNextMove () = 
    let move = rnd.Next 2
    if move = 0 then -1 else 1

let getReward state1 state2 =
    if state2=numStates then 1.0 else 0.0

let getInitValueArray () = 
    Array.create numStates 0.5

let getValue vs s = 
    if s < 0 || s > (numStates - 1) then 0.0
    else Array.get vs s

let calcRMS V1 V2 =
    Array.zip V1 V2
    |> Array.averageBy (fun (x, y) -> pown (x - y) 2)
    |> Math.Sqrt

let rec getValueArray strategy =
    match strategy with
    | TD(_, vs, _) -> vs
    | TDNStepsOnline(_, vs, _, _, _) -> vs
    | TDNStepsOffline(_, vs, _, _, _, _) -> vs
    | LambdaReturn(_, _, _, vs, _, _) -> vs
    | TDLambda (_, _, _, vs, _, _) -> vs
    | ConstantAlphaMC(_, vs, _, _) -> vs
    | Batch(strategy, _, _) -> getValueArray strategy

let rec calcRMSStrategy strategy =
    let refArray = Seq.map (fun v -> v / (float (numStates+1))) (seq { 1.0 .. (float numStates) }) |> Seq.toArray
    calcRMS refArray (getValueArray strategy)

let createTDStrategy alpha = TD(alpha, getInitValueArray(), numStates / 2)

let createTDNStepsOnlineStrategy alpha numSteps = TDNStepsOnline(alpha, getInitValueArray(), numSteps, [numStates / 2], [])

let createLambdaReturnStrategy alpha lambda isOnline = LambdaReturn(alpha, lambda, isOnline, getInitValueArray(), [numStates / 2], [])

let createTDLambdaStrategy alpha lambda useAccTrace = TDLambda(alpha, lambda, useAccTrace, getInitValueArray(), Array.create numStates 0.0, numStates / 2)

let createMCStrategy alpha = ConstantAlphaMC(alpha, getInitValueArray(), [numStates / 2], 0.0)

let createBatchStrategy strategy = Batch(strategy, [[]], [[]])

let rec initStrategy strategy = 
    match strategy with
    | TD(alpha, vs, _) -> TD(alpha, vs, numStates / 2)
    | TDNStepsOnline(alpha, vs, numSteps, _, _) -> TDNStepsOnline(alpha, vs, numSteps, [numStates / 2], [])
    | TDNStepsOffline(alpha, vs, numSteps, _, _, _) -> TDNStepsOffline(alpha, vs, numSteps, [numStates / 2], [], Array.create numStates 0.0)
    | LambdaReturn(alpha, lambda, isOnline, vs, _, _) -> LambdaReturn(alpha, lambda, isOnline, vs, [numStates / 2], [])
    | TDLambda(alpha, lambda, useAccTrace, vs, es, _) -> TDLambda(alpha, lambda, useAccTrace, vs, es, numStates / 2)
    | ConstantAlphaMC(alpha, vs, _, _) -> ConstantAlphaMC(alpha, vs, [numStates / 2], 0.0)
    | Batch(strategy, episodeList, rewardList) -> Batch(initStrategy strategy, episodeList, rewardList)
    
let addElementToInner list elem = 
    match list with
    | hd::tl -> (elem::hd)::tl
    | _ -> [[elem]]

let getNextValues numSteps currValues newValue = 
    List.append (if (List.length currValues) = numSteps then List.tail currValues else currValues) [newValue]
   
let updateStateTransition strategy state reward = 
    match strategy with
    | TD(alpha, vs, currState) -> 
        let vOld = getValue vs currState
        let _ = Array.set vs currState (vOld + alpha * (reward + (getValue vs state) - vOld))
        TD(alpha, vs, state)
    | TDNStepsOnline(alpha, vs, numSteps, currStates, currRewards) -> 
        let nextStates = getNextValues numSteps currStates state
        let nextRewards = getNextValues numSteps currRewards reward
        let _ = if (List.length currStates) < numSteps then ()
                else
                    let updateState = List.head currStates
                    let vOld = getValue vs updateState
                    let returns = (List.sum nextRewards) + getValue vs state
                    Array.set vs updateState (vOld + alpha * (returns - vOld))
        TDNStepsOnline(alpha, vs, numSteps, nextStates, nextRewards)
    | TDNStepsOffline(alpha, vs, numSteps, currStates, currRewards, vs_inc) -> 
        let nextStates = getNextValues numSteps currStates state
        let nextRewards = getNextValues numSteps currRewards reward
        let _ = if (List.length currStates) < numSteps then ()
                else
                    let updateState = List.head currStates
                    let vOld = getValue vs updateState
                    let returns = (List.sum nextRewards) + getValue vs state
                    Array.set vs_inc updateState ((Array.get vs_inc updateState) + alpha * (returns - vOld))
        TDNStepsOffline(alpha, vs, numSteps, nextStates, nextRewards, vs_inc)
    | LambdaReturn(alpha, lambda, isOnline, vs, currStates, currRewards) ->
        let nextStates = state::currStates
        let nextRewards = reward::currRewards
        LambdaReturn(alpha, lambda, isOnline, vs, nextStates, nextRewards)
    | TDLambda(alpha, lambda, useAccTrace, vs, es, currState) ->
        let delta = reward + (Array.get vs state) - (Array.get vs currState)
        let _ = if useAccTrace then Array.set es currState ((Array.get es currState) + 1.0) // accumulating trace
                else Array.set es currState 1.0 // replacing trace
        let _ = Array.mapi (fun i v -> Array.set vs i (v + alpha * delta * (Array.get es i))) vs
        let _ = Array.mapi (fun i v -> Array.set es i (lambda * v)) es
        TDLambda(alpha, lambda, useAccTrace, vs, es, state) 
    | ConstantAlphaMC(alpha, vs, episode, totalReward) -> ConstantAlphaMC(alpha, vs, state::episode, totalReward + reward)
    | Batch(strategy, episodeList, rewardList) -> Batch(strategy, addElementToInner episodeList state, addElementToInner rewardList reward)

let mcUpdate alpha vs reward dummy s =
    let vOld = getValue vs s
    Array.set vs s (vOld + alpha * (reward - vOld))

let reverseFirstElement list = 
    match list with
    | hd::tl -> (List.rev hd)::tl
    | [] -> []

let rec endEpisode strategy = 
    match strategy with
    | TD(alpha, vs, currState) -> initStrategy strategy
    | TDNStepsOnline(alpha, vs, numSteps, currStates, currRewards) -> 
        let states = List.rev (List.tail (List.rev currStates))
        let _ = List.foldBack2 (fun state reward returns ->
                                    let vOld = getValue vs state
                                    let nextReturns = returns + reward
                                    let _ = Array.set vs state (vOld + alpha * (nextReturns - vOld))
                                    nextReturns
                               ) states currRewards 0.0
        initStrategy strategy
    | TDNStepsOffline(alpha, vs, numSteps, currStates, currRewards, vs_inc) -> 
        let states = List.rev (List.tail (List.rev currStates))
        let _ = List.foldBack2 (fun state reward returns ->
                                    let vOld = getValue vs state
                                    let nextReturns = returns + reward
                                    let _ = Array.set vs_inc state (alpha * (nextReturns - vOld))
                                    nextReturns
                               ) states currRewards 0.0
        let _ = Array.iteri (fun idx value -> Array.set vs idx (value + (Array.get vs_inc idx))) vs
        initStrategy strategy
    | LambdaReturn(alpha, lambda, isOnline, vs, currStates, currRewards) ->
        let currRewardsFwd = List.rev (List.tail currRewards)
        let currStatesFwd = List.rev (List.tail currStates)
        let lastReward = List.head currRewards
        let stateUpdates = 
            List.mapi (fun idx value -> 
                       let remainingRewards = currRewardsFwd |> List.toSeq |> Seq.skip idx 
                       let nStepRewards = remainingRewards
                                          |> Seq.mapi (fun i v -> v * (lambda ** (float i)))
                                          |> Seq.sum
                       let lambdaEnd = lambda ** (float (Seq.length remainingRewards))
                       (1.0 - lambda) * nStepRewards + lambdaEnd * lastReward) currStatesFwd
        let _ = 
            if isOnline then
                List.map2 (fun s v -> 
                           let vOld = Array.get vs s
                           Array.set vs s (vOld + alpha * (v - vOld))) currStatesFwd stateUpdates
            else 
                let vs_inc = Array.create numStates 0.0
                let _ = List.map2 (fun s v ->
                                   let vOld = Array.get vs s
                                   Array.set vs_inc s ((Array.get vs_inc s) + alpha * (v - vOld))) currStatesFwd stateUpdates
                Array.mapi (fun i v ->
                            Array.set vs i ((Array.get vs i) + v)) vs_inc |> Array.toList
        initStrategy strategy
    | TDLambda(_, _, _, _, _, _) -> initStrategy strategy                         
    | ConstantAlphaMC(alpha, vs, headEpisode::tailEpisode, totalReward) -> 
        let episodeActual = List.rev tailEpisode
        let _ = List.fold (mcUpdate alpha vs totalReward) () episodeActual
        initStrategy strategy
    | ConstantAlphaMC(alpha, vs, episode, totalReward) -> initStrategy (ConstantAlphaMC(alpha, vs, episode, totalReward))
    | Batch(strategy, episodeList, rewardList) -> 
        let episodeListTf = reverseFirstElement episodeList
        let rewardListTf = reverseFirstElement rewardList
        let rec runBatchUpdate st =
            let (stNext, maxDiff) = simulateStrategy strategy episodeListTf rewardListTf
            let convTol = 1e-6
            if maxDiff < convTol then stNext else runBatchUpdate stNext
        let strategyNext = runBatchUpdate strategy
        Batch(strategyNext, []::episodeListTf, []::rewardListTf)

and simulateStrategy strategy episodes rewards =
    let Vorig = Array.copy (getValueArray strategy)
    let simulateFoldFun st ep rew = 
        endEpisode (List.fold2 (fun s e r -> updateStateTransition s e r) (initStrategy st) ep rew)
    let strategyNext = List.fold2 simulateFoldFun strategy episodes rewards
    let inc = Array.fold2 (fun x v1 v2 -> max x (abs (v1-v2))) 0.0 Vorig (getValueArray strategy)
    if inc < 1e1 then (strategyNext, inc) else failwith "algorithm is diverging, decrease alpha"

let runOneEpisode strategy = 
    let rec episodeHelper strategy state = 
        let move = getNextMove()
        let nextState = state + move
        let hasReachedEnd = nextState > (numStates - 1) || nextState < 0
        let reward = if nextState > (numStates - 1) then 1.0 else 0.0
        let updatedStrategy = updateStateTransition strategy nextState reward
        if hasReachedEnd then endEpisode updatedStrategy 
        else episodeHelper updatedStrategy nextState
    episodeHelper strategy 2

let rec runEpisodes (strategy, rmsList) numEpisodes = 
    if numEpisodes=0 then (strategy, List.rev rmsList)
    else
        let nextStrategy = runOneEpisode strategy
        runEpisodes (nextStrategy, (calcRMSStrategy nextStrategy)::rmsList) (numEpisodes-1)

let numEpisodes = 100

let (td0_15, rmsTD0_15) = runEpisodes (createTDStrategy 0.15, List.empty) numEpisodes
let (td0_1, rmsTD0_1) = runEpisodes (createTDStrategy 0.1, List.empty) numEpisodes
let (td0_05, rmsTD0_05) = runEpisodes (createTDStrategy 0.05, List.empty) numEpisodes
let (mc0_01, rmsMC0_01) = runEpisodes (createMCStrategy 0.01, List.empty) numEpisodes
let (mc0_02, rmsMC0_02) = runEpisodes (createMCStrategy 0.02, List.empty) numEpisodes
let (mc0_03, rmsMC0_03) = runEpisodes (createMCStrategy 0.03, List.empty) numEpisodes
let (mc0_04, rmsMC0_04) = runEpisodes (createMCStrategy 0.04, List.empty) numEpisodes
let (mcBatch0_01, rmsMCBatch0_01) = runEpisodes (createBatchStrategy (createMCStrategy 0.01), List.empty) numEpisodes
let (tdBatch0_01, rmsTDBatch0_01) = runEpisodes (createBatchStrategy (createTDStrategy 0.01), List.empty) numEpisodes

let rmsSeriesTD0_05 = Series.ofObservations (Seq.zip (seq {1 .. numEpisodes }) rmsTD0_05)
Chart.Line(rmsSeriesTD0_05)

Chart.Combine(
    [ Chart.Line(rmsTD0_05)
      Chart.Line(rmsTD0_1)
      Chart.Line(rmsTD0_15)
      Chart.Line(rmsMC0_01)
      Chart.Line(rmsMC0_02)
      Chart.Line(rmsMC0_03)
      Chart.Line(rmsMC0_04) ])
