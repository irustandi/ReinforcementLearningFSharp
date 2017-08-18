#load "../packages/FsLab/FsLab.fsx"

open System
open Deedle
open XPlot.GoogleCharts
open XPlot.GoogleCharts.Deedle

type Strategy =
    | TD of float * float [] * int
    | ConstantAlphaMC of float * float [] * int list * float
    | Batch of Strategy * int list list * float list list

let rnd = Random()

let getNextMove () = 
    let move = rnd.Next 2
    if move = 0 then -1 else 1

let getReward state1 state2 =
    if state2=5 then 1.0 else 0.0

let getInitValueArray () = 
    Array.create 5 0.5

let getValue Vs s = 
    if s < 0 || s > 4 then 0.0
    else Array.get Vs s

let calcRMS V1 V2 =
    Array.zip V1 V2
    |> Array.averageBy (fun (x, y) -> pown (x - y) 2)
    |> Math.Sqrt

let rec calcRMSStrategy strategy =
    let refArray = Seq.map (fun v -> v / 6.0) (seq { 1.0 .. 5.0 }) |> Seq.toArray
    match strategy with
    | TD(_, Vs, _) -> calcRMS refArray Vs
    | ConstantAlphaMC(_, Vs, _, _) -> calcRMS refArray Vs
    | Batch(strategy, _, _) -> calcRMSStrategy strategy

let rec getValueArray strategy =
    match strategy with
    | TD(_, Vs, _) -> Vs
    | ConstantAlphaMC(_, Vs, _, _) -> Vs
    | Batch(strategy, _, _) -> getValueArray strategy

let createTDStrategy alpha = TD(alpha, getInitValueArray(), 2)

let createMCStrategy alpha = ConstantAlphaMC(alpha, getInitValueArray(), [2], 0.0)

let createBatchStrategy strategy = Batch(strategy, [[]], [[]])

let rec initStrategy strategy = 
    match strategy with
    | TD(alpha, Vs, state) -> TD(alpha, Vs, 2)
    | ConstantAlphaMC(alpha, Vs, episode, totalReward) -> ConstantAlphaMC(alpha, Vs, [2], 0.0)
    | Batch(strategy, episodeList, rewardList) -> Batch(initStrategy strategy, episodeList, rewardList)

let addElementToInner list elem = 
    match list with
    | hd::tl -> (elem::hd)::tl
    | _ -> [[elem]]
   
let updateStateTransition strategy state reward = 
    match strategy with
    | TD(alpha, Vs, currState) -> 
        let vOld = getValue Vs currState
        let _ = Array.set Vs currState (vOld + alpha * (reward + (getValue Vs state) - vOld))
        TD(alpha, Vs, state)
    | ConstantAlphaMC(alpha, Vs, episode, totalReward) -> ConstantAlphaMC(alpha, Vs, state::episode, totalReward + reward)
    | Batch(strategy, episodeList, rewardList) -> Batch(strategy, addElementToInner episodeList state, addElementToInner rewardList reward)

let mcUpdate alpha Vs reward dummy s =
    let vOld = getValue Vs s
    Array.set Vs s (vOld + alpha * (reward - vOld))

let reverseFirstElement list = 
    match list with
    | hd::tl -> (List.rev hd)::tl
    | [] -> []
       
let rec endEpisode strategy = 
    match strategy with
    | TD(alpha, Vs, currState) -> initStrategy (TD(alpha, Vs, currState))
    | ConstantAlphaMC(alpha, Vs, headEpisode::tailEpisode, totalReward) -> 
        let episodeActual = List.rev tailEpisode
        let _ = List.fold (mcUpdate alpha Vs totalReward) () episodeActual
        initStrategy (ConstantAlphaMC(alpha, Vs, headEpisode::tailEpisode, totalReward))
    | ConstantAlphaMC(alpha, Vs, episode, totalReward) -> initStrategy (ConstantAlphaMC(alpha, Vs, episode, totalReward))
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
        let hasReachedEnd = nextState > 4 || nextState < 0
        let reward = if nextState>4 then 1.0 else 0.0
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
