#load "../packages/FsLab/FsLab.fsx"

open System
open MathNet.Numerics.Random
open MathNet.Numerics.Distributions
open XPlot.GoogleCharts

type ActionSelection = 
    | EpsilonGreedy of float * float [] * float [] * float []
    | Softmax of float * float [] * float [] * float []
    | EpsilonGreedyIncremental of float * float * int [] * float []
    | EpsilonGreedyConstantAlpha of float * float * float []
    | ReinforcementComparison of float * float * bool * float * float []
    | Pursuit of float * int [] * float [] * float []

let rnd = Random()
let mt = MersenneTwister()
let n = 10
let nTasks = 2000
let nPlays = 1000

let initAlgorithm algo = 
    match algo with
    | EpsilonGreedy(eps, _, _, _) -> EpsilonGreedy(eps, Array.create n 0.0, Array.create n 0.0, Array.create n 0.0)
    | Softmax(tau, _, _, _) -> Softmax(tau, Array.create n 0.0, Array.create n 0.0, Array.create n 0.0)
    | EpsilonGreedyIncremental(eps, initValue, _, _) -> EpsilonGreedyIncremental(eps, initValue, Array.create n 0, Array.create n initValue)
    | EpsilonGreedyConstantAlpha(eps, alpha, _) -> EpsilonGreedyConstantAlpha(eps, alpha, Array.create n 0.0)
    | ReinforcementComparison(alpha, beta, doEnhanced, _, _) -> ReinforcementComparison(alpha, beta, doEnhanced, 0.0, Array.create n (1.0/(float n)))
    | Pursuit(beta, _, _, _) -> Pursuit(beta, Array.create n 0, Array.create n 0.0, Array.create n (1.0/(float n)))

let sigmaRW = 30.0
let qValuesRef = Array2D.init nTasks n (fun _ _ -> Normal.Sample(mt, 0.0, 1.0))

let sampleArm probs = Multinomial.Sample(mt, probs, 1) |> Array.findIndex (fun v -> v=1)

let getAction algo = 
    match algo with
    | EpsilonGreedy(eps, q, _, _) -> if rnd.NextDouble() < eps then rnd.Next(n) 
                                     else Seq.maxBy (fun armIdx -> Array.get q armIdx) (seq { 0 .. (n-1) })
    | Softmax(tau, q, _, _) -> let qovertau = Array.map (fun v -> Math.Exp(v / tau)) q
                               let probs = Array.map (fun v -> v / (Array.sum qovertau)) qovertau
                               sampleArm probs
    | EpsilonGreedyIncremental(eps, _, _, q) -> 
        if rnd.NextDouble() < eps then rnd.Next(n)
        else Seq.maxBy (fun armIdx -> Array.get q armIdx) (seq { 0 .. (n-1) })
    | EpsilonGreedyConstantAlpha(eps, _, q) ->
        if rnd.NextDouble() < eps then rnd.Next(n)
        else Seq.maxBy (fun armIdx -> Array.get q armIdx) (seq { 0 .. (n-1) })
    | ReinforcementComparison(_, _, _, _, ps) ->
        sampleArm (ps |> Array.map Math.Exp)       
    | Pursuit(_, _, _, ps) ->
        sampleArm ps

let updateAlgorithm algo armIdx reward =
    match algo with
    | EpsilonGreedy(eps, q, qsum, qnum) ->
        let qsumNew = qsum |> Array.mapi (fun i q -> if i=armIdx then q+reward else q)
        let qnumNew = qnum |> Array.mapi (fun i q -> if i=armIdx then q+1.0 else q)
        let qNew = q |> Array.mapi (fun i x -> if i=armIdx then (qsum.[i] / qnum.[i]) else x)
        EpsilonGreedy(eps, qNew, qsumNew, qnumNew)
    | Softmax(tau, q, qsum, qnum) -> 
        let qsumNew = qsum |> Array.mapi (fun i q -> if i=armIdx then q+reward else q)
        let qnumNew = qnum |> Array.mapi (fun i q -> if i=armIdx then q+1.0 else q)
        let qNew = q |> Array.mapi (fun i x -> if i=armIdx then (qsum.[i] / qnum.[i]) else x)
        Softmax(tau, qNew, qsumNew, qnumNew)
    | EpsilonGreedyIncremental(eps, initValue, ks, qs) ->
        let kNew = (Array.get ks armIdx) + 1
        let alpha = 1.0 / (float kNew)
        let qNew = (1.0 - alpha) * (Array.get qs armIdx) + alpha * reward
        let ksNew = ks |> Array.mapi (fun i k -> if i=armIdx then kNew else k)
        let qsNew = qs |> Array.mapi (fun i q -> if i=armIdx then qNew else q)
        EpsilonGreedyIncremental(eps, initValue, ksNew, qsNew)
    | EpsilonGreedyConstantAlpha(eps, alpha, qs) ->
        let qNew = (1.0 - alpha) * (Array.get qs armIdx) + alpha * reward
        let qsNew = qs |> Array.mapi (fun i q -> if i=armIdx then qNew else q)
        EpsilonGreedyConstantAlpha(eps, alpha, qsNew)
    | ReinforcementComparison(alpha, beta, doEnhanced, refReward, ps) ->
        let rewardDiff = reward - refReward
        let probIncrementRaw = beta * rewardDiff
        let probIncrement = 
            if doEnhanced then
                let expPs = ps |> Array.map Math.Exp
                probIncrementRaw + (1.0 - (Array.get expPs armIdx) / (Array.sum expPs))
            else probIncrementRaw
        let psNew = Array.mapi (fun i p -> 
                                       if i=armIdx then p + probIncrement 
                                       else p) ps
        let refRewardNew = refReward + alpha * rewardDiff
        ReinforcementComparison(alpha, beta, doEnhanced, refRewardNew, psNew)
    | Pursuit(beta, ks, qs, ps) ->
        let kNew = (Array.get ks armIdx) + 1
        let alpha = 1.0 / (float kNew)
        let qNew = (1.0 - alpha) * (Array.get qs armIdx) + alpha * reward
        let ksNew = ks |> Array.mapi (fun i k -> if i=armIdx then kNew else k)
        let qsNew = qs |> Array.mapi (fun i q -> if i=armIdx then qNew else q)
        let nextGreedyArmIdx = Seq.maxBy (fun armIdx -> Array.get qs armIdx) (seq { 0 .. (n-1) })
        let psNew = ps |> Array.mapi (fun i p ->
                                          if i=nextGreedyArmIdx then (1.0 - beta) * p + beta
                                          else (1.0 - beta) * p)
        Pursuit(beta, ksNew, qsNew, psNew)


let processTask doRandomWalk qValues algo allRewards pickedMaxAction taskIdx =
    let bestArmIdx = Seq.maxBy (fun armIdx -> Array2D.get qValues taskIdx armIdx) (seq { 0 .. (n-1) })
    // let _ = printfn "task %d; best arm %d" taskIdx bestArmIdx
    let processPlay playAlgo playIdx =
        let armIdx = getAction playAlgo
        let _ = if armIdx = bestArmIdx then Array.set (Array.get pickedMaxAction playIdx) taskIdx true else ()
        let reward = (Array2D.get qValues taskIdx armIdx) + Normal.Sample(mt, 0.0, 1.0)
        let _ = if doRandomWalk then ignore (Array.map (fun arm -> 
                                                           let qVal = Array2D.get qValues taskIdx arm
                                                           Array2D.set qValues taskIdx arm (qVal + sigmaRW * Normal.Sample(mt, 0.0, 1.0))) (seq { 0 .. (n-1) } |> Seq.toArray))
                else ()
        // let _ = printfn "task: %d; play: %d; reward: %g" taskIdx playIdx reward
        let _ = Array.set (Array.get allRewards playIdx) taskIdx reward
        updateAlgorithm playAlgo armIdx reward
    let _ = Seq.fold processPlay (initAlgorithm algo) (seq { 0 .. (nPlays-1) })
    ()

let processAlgorithm doRandomWalk algo =
    let qValues = if doRandomWalk then Array2D.create nTasks n 1.0 
                  else qValuesRef
    let allRewards = Array.init nPlays (fun _ -> Array.create nTasks 0.0)
    let pickedMaxAction = Array.init nPlays (fun _ -> Array.create nTasks false)
    let _ = Array.Parallel.map (processTask doRandomWalk qValues algo allRewards pickedMaxAction) (seq { 0 .. (nTasks-1) } |> Seq.toArray)
    (allRewards, pickedMaxAction)

let processEpsilonGreedy eps = processAlgorithm false (EpsilonGreedy(eps, Array.empty, Array.empty, Array.empty))
let processSoftmax tau = processAlgorithm false (Softmax(tau, Array.empty, Array.empty, Array.empty))
let processEpsilonGreedyIncremental eps initValue = processAlgorithm false (EpsilonGreedyIncremental(eps, initValue, Array.empty, Array.empty))
let processRWEpsilonGreedyIncremental eps = processAlgorithm true (EpsilonGreedyIncremental(eps, 0.0, Array.empty, Array.empty))
let processRWEpsilonGreedyConstantAlpha eps alpha = processAlgorithm true (EpsilonGreedyConstantAlpha(eps, alpha, Array.empty))
let processReinforcementComparison alpha beta doEnhanced = processAlgorithm true (ReinforcementComparison(alpha, beta, doEnhanced, 0.0, Array.empty))
let processPursuit beta = processAlgorithm true (Pursuit(beta, Array.empty, Array.empty, Array.empty))

let getOptions title = 
    Options
        ( title = title, 
          legend = Legend(position="bottom"))

let (allRewards_eps_0, pickedMaxAction_eps_0) = processEpsilonGreedy 0.0
let (allRewards_eps_0_01, pickedMaxAction_eps_0_01) = processEpsilonGreedy 0.01
let (allRewards_eps_0_1, pickedMaxAction_eps_0_1) = processEpsilonGreedy 0.1

let avgRewards_eps_0 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_eps_0
let avgRewards_eps_0_01 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_eps_0_01
let avgRewards_eps_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_eps_0_1


[avgRewards_eps_0; avgRewards_eps_0_01; avgRewards_eps_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedy N-Armed Testbed Rewards")
|> Chart.WithLabels ["eps 0"; "eps 0.01"; "eps 0.1"]

let pctMaxActs_eps_0 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_eps_0
let pctMaxActs_eps_0_01 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_eps_0_01
let pctMaxActs_eps_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_eps_0_1

[pctMaxActs_eps_0; pctMaxActs_eps_0_01; pctMaxActs_eps_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedy N-Armed Testbed % Max Action")
|> Chart.WithLabels ["eps 0"; "eps 0.01"; "eps 0.1"]

let (allRewards_tau_0_1, pickedMaxAction_tau_0_1) = processSoftmax 0.1
let (allRewards_tau_0_25, pickedMaxAction_tau_0_25) = processSoftmax 0.25
let (allRewards_tau_0_5, pickedMaxAction_tau_0_5) = processSoftmax 0.5
let avgRewards_tau_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_tau_0_1
let avgRewards_tau_0_25 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_tau_0_25
let avgRewards_tau_0_5 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_tau_0_5

[avgRewards_tau_0_1; avgRewards_tau_0_25; avgRewards_tau_0_5] 
|> Chart.Line
|> Chart.WithOptions (getOptions "Softmax N-Armed Testbed Rewards")
|> Chart.WithLabels ["tau 0.1"; "tau 0.25"; "tau 0.5"]

let pctMaxActs_tau_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_tau_0_1
let pctMaxActs_tau_0_25 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_tau_0_25
let pctMaxActs_tau_0_5 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_tau_0_5

[pctMaxActs_tau_0_1; pctMaxActs_tau_0_25; pctMaxActs_tau_0_5] 
|> Chart.Line
|> Chart.WithOptions (getOptions "Softmax N-Armed Testbed % Max Action")
|> Chart.WithLabels ["tau 0.1"; "tau 0.25"; "tau 0.5"]

let (allRewards_epsInc_0, pickedMaxAction_epsInc_0) = processEpsilonGreedyIncremental 0.0 0.0
let (allRewards_epsInc_0_01, pickedMaxAction_epsInc_0_01) = processEpsilonGreedyIncremental 0.01 0.0
let (allRewards_epsInc_0_1, pickedMaxAction_epsInc_0_1) = processEpsilonGreedyIncremental 0.1 0.0

let avgRewards_epsInc_0 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_epsInc_0
let avgRewards_epsInc_0_01 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_epsInc_0_01
let avgRewards_epsInc_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_epsInc_0_1

[avgRewards_epsInc_0; avgRewards_epsInc_0_01; avgRewards_epsInc_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedyIncremental N-Armed Testbed Rewards")
|> Chart.WithLabels ["eps 0"; "eps 0.01"; "eps 0.1"]

let pctMaxActs_epsInc_0 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_epsInc_0
let pctMaxActs_epsInc_0_01 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_epsInc_0_01
let pctMaxActs_epsInc_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_epsInc_0_1

[pctMaxActs_epsInc_0; pctMaxActs_epsInc_0_01; pctMaxActs_epsInc_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedyIncremental N-Armed Testbed % Max Action")
|> Chart.WithLabels ["eps 0"; "eps 0.01"; "eps 0.1"]

let (allRewards_RW_epsInc_0_1, pickedMaxAction_RW_epsInc_0_1) = processRWEpsilonGreedyIncremental 0.1
let (allRewards_RW_epsAlp_0_1, pickedMaxAction_RW_epsAlp_0_1) = processRWEpsilonGreedyConstantAlpha 0.1 0.1
let (allRewards_epsIncOpt_0_1, pickedMaxAction_epsIncOpt_0_1) = processEpsilonGreedyIncremental 0.1 5.0
let (allRewards_reinfComp, pickedMaxAction_reinfComp) = processReinforcementComparison 0.1 0.1 false
let (allRewards_reinfComp_enhanced, pickedMaxAction_reinfComp_enhanced) = processReinforcementComparison 0.1 0.1 true
let (allRewards_pursuit, pickedMaxAction_pursuit) = processPursuit 0.1

let avgRewards_RW_epsInc_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_RW_epsInc_0_1
let avgRewards_RW_epsAlp_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_RW_epsAlp_0_1
let avgRewards_epsIncOpt_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_epsIncOpt_0_1
let avgRewards_reinfComp = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_reinfComp
let avgRewards_reinfComp_enhanced = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_reinfComp_enhanced
let avgRewards_pursuit = Array.Parallel.mapi (fun i arr -> (i, Array.average arr)) allRewards_pursuit

let pctMaxActs_RW_epsInc_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_RW_epsInc_0_1
let pctMaxActs_RW_epsAlp_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_RW_epsAlp_0_1
let pctMaxActs_epsIncOpt_0_1 = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_epsIncOpt_0_1
let pctMaxActs_reinfComp = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_reinfComp
let pctMaxActs_reinfComp_enhanced = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_reinfComp_enhanced
let pctMaxActs_pursuit = Array.Parallel.mapi (fun i arr -> (i, Array.averageBy (fun v -> if v then 1.0 else 0.0) arr)) pickedMaxAction_pursuit

[avgRewards_RW_epsInc_0_1; avgRewards_RW_epsAlp_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedy Nonstationary N-Armed Testbed Rewards")
|> Chart.WithLabels ["incremental"; "constant alpha"]

[pctMaxActs_RW_epsInc_0_1; pctMaxActs_RW_epsAlp_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedy Nonstationary N-Armed Testbed % Max Action")
|> Chart.WithLabels ["incremental"; "constant alpha"]

[avgRewards_epsInc_0_1; avgRewards_epsIncOpt_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedy N-Armed Testbed Rewards")
|> Chart.WithLabels ["baseline"; "optimistic"]

[pctMaxActs_epsInc_0_1; pctMaxActs_epsIncOpt_0_1] 
|> Chart.Line
|> Chart.WithOptions (getOptions "EpsilonGreedy N-Armed Testbed % Max Action")
|> Chart.WithLabels ["baseline"; "optimistic"]

[avgRewards_epsInc_0_1; avgRewards_reinfComp; avgRewards_reinfComp_enhanced; avgRewards_pursuit] 
|> Chart.Line
|> Chart.WithOptions (getOptions "N-Armed Testbed Rewards")
|> Chart.WithLabels ["baseline"; "reinforcement comparison"; "reinforcement comparison (enhanced)"; "pursuit"]

[pctMaxActs_epsInc_0_1; pctMaxActs_reinfComp; pctMaxActs_reinfComp_enhanced; pctMaxActs_pursuit] 
|> Chart.Line
|> Chart.WithOptions (getOptions "N-Armed Testbed % Max Action")
|> Chart.WithLabels ["baseline"; "reinforcement comparison"; "reinforcement comparison (enhanced)"; "pursuit"]
