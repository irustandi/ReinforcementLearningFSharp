let p = 0.45

let backupAction V s a =
    p * (Array.get V (s + a)) + (1.0-p) * (Array.get V (s-a))

let findMaxForState V delta s = 
    let v = Array.get V s
    let vNew = Seq.max (Seq.map (fun a -> backupAction V s a) (seq { 1 .. (min s (100-s)) }))
    let dummy = Array.set V s vNew
    let absDiff = abs (v - vNew)
    if absDiff > delta then absDiff else delta

let valueIteration epsilon = 
    let V = Array.create 101 0.0
    let dummy = Array.set V 100 1.0
    let rec valueIterationHelper V = 
        let currDelta = Seq.fold (findMaxForState V) 0.0 (seq { 1 .. 99 })
        if currDelta < epsilon then V else valueIterationHelper V
    valueIterationHelper V

let policy V s epsilon = 
    let foldFun (bv, ba) a =
        let currValue = backupAction V s a
        if currValue > bv + epsilon then (currValue, a) else (bv, ba)
    Seq.fold foldFun (-1.0, -1) (seq { 1 .. (min s (100-s)) })