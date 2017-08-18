open System
open System.Numerics

let maxNumCars = 20
let maxNumCarsToMove = 5
let lambda1Req = 3.0
let lambda1Ret = 3.0
let lambda2Req = 4.0
let lambda2Ret = 2.0
let gamma = 0.9
let probThres = 1e-6
let rentalRate = 10.0
let moveCost = 2
let theta = 1e-7
let epsilon = 1e-10

let rec factorial n = 
    if n=BigInteger.Zero then BigInteger.One else BigInteger.Multiply(n, factorial (BigInteger.Subtract(n, BigInteger.One)))

let poisson (n:int) lambda = 
    Math.Exp (-lambda) * Math.Pow(lambda, float n) / float (factorial (BigInteger(n)))

let loadRewardVec lambda = 
    let rec loadRewardVecHelper R lambda numRequests =
        let requestProb = poisson numRequests lambda
        if requestProb <= probThres then R 
        else
            let Rnext = Array.mapi (fun n value -> value + rentalRate * requestProb * (float (min numRequests n))) R
            loadRewardVecHelper Rnext lambda (numRequests + 1)
    loadRewardVecHelper (Array.create 26 0.0) lambda 0

let loadProbMat lambdaReq lambdaRet = 
    let rec loadProbMatReturnHelper P lambdaRet reqProb numRequests numReturns = 
        let returnProb = poisson numReturns lambdaRet
        if returnProb <= probThres then P
        else
            let mapFun n =
                let numFilledRequests = min numRequests n
                let numNew = max 0 (min maxNumCars (n + numReturns - numFilledRequests))
                let oldValue = Array2D.get P n numNew
                Array2D.set P n numNew (oldValue + reqProb * returnProb)
            let _ = Array.map mapFun (seq { 0 .. 25 } |> Seq.toArray)
            loadProbMatReturnHelper P lambdaRet reqProb numRequests (numReturns + 1)
    let rec loadProbMatHelper P lambdaReq lambdaRet numRequests = 
        let requestProb = poisson numRequests lambdaReq
        if requestProb <= probThres then P
        else
            let Pnext = loadProbMatReturnHelper P lambdaRet requestProb numRequests 0
            loadProbMatHelper Pnext lambdaReq lambdaRet (numRequests + 1)
    loadProbMatHelper (Array2D.create 26 21 0.0) lambdaReq lambdaRet 0
 
let rewardVec1 = loadRewardVec lambda1Req
let rewardVec2 = loadRewardVec lambda2Req
let probMat1 = loadProbMat lambda1Req lambda2Ret
let probMat2 = loadProbMat lambda2Req lambda2Ret

let backupAction n1 n2 a vMat = 
    let action = max (-n2) (min a n1)
    let actionFinal = max (-maxNumCarsToMove) (min maxNumCarsToMove action)
    let cost = -moveCost * (Math.Abs actionFinal)
    let n1Start = n1 - actionFinal
    let n2Start = n2 + actionFinal
    let foldFun value (n1, n2) =
        let probTerm = (Array2D.get probMat1 n1Start n1) * (Array2D.get probMat2 n2Start n2)
        let rewardTerm = (Array.get rewardVec1 n1Start) + (Array.get rewardVec2 n2Start) + gamma * (Array2D.get vMat n1 n2)
        value + probTerm * rewardTerm
    let idxList = [ for x in 0 .. 20 do
                    for y in 0 .. 20 do
                    yield x,y ]
    (List.fold foldFun 0.0 idxList) + (float cost)

let getBestAction n1 n2 vMat =
    let foldFun (currValue, currAction) action =
        let thisValue = backupAction n1 n2 action vMat
        if thisValue > currValue + epsilon then (thisValue, action) else (currValue, currAction)
    let startRange = max (-maxNumCarsToMove) (-n2)
    let endRange = (min maxNumCarsToMove n1)
    let (bestValue, bestAction) = Array.fold foldFun (-1.0, 0) (seq { startRange .. endRange } |> Seq.toArray)
    bestAction

let policyEvaluation vMat policyMat = 
    let rec policyEvalHelper vMat policyMat delta =
        if delta < theta then ()
        else
            let foldFun deltaVal (n1, n2) =
                let vOld = Array2D.get vMat n1 n2
                let action = Array2D.get policyMat n1 n2
                let vNew = backupAction n1 n2 action vMat
                let _ = Array2D.set vMat n1 n2 vNew
                max deltaVal (Math.Abs (vOld - vNew))
            let idxList = [ for x in 0 .. 20 do
                            for y in 0 .. 20 do
                            yield x,y ]
            let nextDelta = List.fold foldFun 0.0 idxList
            policyEvalHelper vMat policyMat nextDelta
    policyEvalHelper vMat policyMat 1.0

let improvePolicy policyMat vMat = 
    let idxList = [ for x in 0 .. 20 do
                    for y in 0 .. 20 do
                    yield x,y ]
    let foldFun improved (n1, n2) =
        let prevAction = Array2D.get policyMat n1 n2
        let bestAction = getBestAction n1 n2 vMat
        let _ = Array2D.set policyMat n1 n2 bestAction
        if bestAction <> prevAction then true else improved
    List.fold foldFun false idxList

let vMat = Array2D.create 21 21 0.0
let policyMat = Array2D.create 21 21 0

let rec runPolicyIteration policyImproved (count:int) = 
    let _ = count |> Console.WriteLine
    if not policyImproved then () 
    else
        let _ = policyEvaluation vMat policyMat
        let nextPolicyImproved = improvePolicy policyMat vMat
        runPolicyIteration nextPolicyImproved (count + 1)

runPolicyIteration true 0

