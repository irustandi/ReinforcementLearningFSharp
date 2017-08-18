open System

let alpha = 0.2
let numPatches = 100
let widths = [|3; 9; 27|]
let numTrainingAll = [|10; 40; 160; 640; 2560; 10240|]

let targetFun x = if x >= 0.4 && x < 0.6 then 1.0 else 0.0

let withinPatch width i x = (Math.Abs (x - (float i)/(float numPatches))) <= ((float width) / (2.0 * (float numPatches)))

let linearApproxFun width w x =
     seq { for i = 1 to numPatches do yield (i-1) } |> Seq.sumBy (fun i -> if withinPatch width i x then (Array.get w i) else 0.0)

let rng = Random()
let trainFun width w = 
    let x = 0.25 + 0.5 * rng.NextDouble()
    let target = targetFun x
    let fVal = linearApproxFun width w x
    let alphaError = alpha / (float width) * (target - fVal)
    seq { for i = 1 to numPatches do yield (i-1) } 
    |> Seq.fold (fun _ i -> if withinPatch width i x then Array.set w i ((Array.get w i) + alphaError) else ()) ()

