namespace TileCoding

module Tiles =
    let rng = System.Random()

    let initializeRandomSeq length = 
        Array.create length 0
        |> Array.map (fun v -> (v <<< 8) ||| (rng.Next() &&& 0xff))

    let rndSeq = initializeRandomSeq 2048

    let hashCoordinates coordinates memorySize =
        let sum = 
            Array.fold2 (fun s v i -> 
                            let index = (v + (449 * i)) % (Array.length rndSeq)
                            s + (int (Array.get rndSeq index))) 
                        0 coordinates (seq { for x=1 to Array.length coordinates do yield x-1 } |> Seq.toArray)
        sum % memorySize

    let getTiles numTilings variables memorySize = 
        let numVariables = Array.length variables
        let numCoordinates = numVariables + 1
        let qstate = Array.create numVariables 0
        let baseArray = Array.create numVariables 0
        let coordinates = Array.create numCoordinates 0

        let _ = Array.mapi 
                    (fun i v -> 
                        let _ = Array.set qstate i (int (v * (float numTilings)))
                        Array.set baseArray i 0) variables
        
        let processTile idx = 
            let _ = 
                Seq.map (fun i -> 
                              let qStateVal = Array.get qstate i
                              let baseVal = Array.get baseArray i
                              let _ = if qStateVal >= baseVal then
                                          Array.set coordinates i (qStateVal - ((qStateVal - baseVal) % numTilings))
                                      else Array.set coordinates i (qStateVal + 1 + ((baseVal - qStateVal - 1) % numTilings) - numTilings)
                              Array.set baseArray i (baseVal + 1 + (2 * i)))
                          (seq {for i = 1 to numVariables do yield i-1 })
            let _ = Array.set coordinates numVariables idx
            hashCoordinates coordinates memorySize
        Seq.map processTile (seq { for i = 1 to numTilings do yield i-1 }) |> Seq.toArray
