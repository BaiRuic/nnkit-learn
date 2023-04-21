import pathlib

types = ["int8_t", "int16_t", "int32_t", "int64_t",
         "uint8_t", "uint16_t", "uint32_t", "uint64_t",
         "_Float32", "_Float64"]

types = ["int32_t", "int64_t",
         "uint32_t", "uint64_t",
         "_Float64"]

def generateOneType(oneTypeName):
    res = []
    # m.def("to_list", to_list<int32_t>);
    for pyFunc, cppFunc in oneTypeName:
        if cppFunc == "":
            cppFunc = pyFunc
        for type in types:
            temp = f'm.def("{pyFunc}", {cppFunc}<{type}>);'
            res.append(temp)
    return res

def generateTwoType(twoTypeName):
    res = []
    # eg. m.def("from_handle", from_handle<uint64_t, uint64_t>);
    for pyFunc, cppFunc in twoTypeName:
        if cppFunc == "":
            cppFunc = pyFunc
        for type1 in types:
            for type2 in types:
                temp = f'm.def("{pyFunc}", {cppFunc}<{type1}, {type2}>);'
                res.append(temp)
    return res

def generateThreeType(threeTypeName):
    res = []
    # eg. m.def("from_handle", from_handle<uint64_t, uint64_t>);
    for pyFunc, cppFunc in threeTypeName:
        if cppFunc == "":
            cppFunc = pyFunc
        for type1 in types:
            for type2 in types:
                for type3 in types:
                    temp = f'm.def("{pyFunc}", {cppFunc}<{type1}, {type2}, {type3}>);'
                    res.append(temp)
    return res

def writeTo(fileName, stringList, prepending_spaces=4):
    try:
        with open(fileName, 'w') as f:
            for s in stringList:
                f.write(" " * prepending_spaces + s + '\n')
    except IOError:
        print("Error: 无法写入文件。")
    except Exception as ex:
        print("Error: ", ex)
    else:
        print("Done")



if __name__ == "__main__":
    fileName = pathlib.Path(__file__).resolve().parent.joinpath("templatedCode.cpp")
    print(fileName)
    oneTypeName = [["from_numpy", ""],
                   ["from_pylist", ""],
                   ["to_numpy", ""],
                   ["to_list", ""],
                   ["fill", "Fill"],
                   ["compact", "Compact"],
                   ["ewise_setitem", "EwiseSetitem"],
                   ["scalar_setitem", "ScalarSetitem"],
                   ["scalar_power", "ScalarPower"],
                   ["ewise_eq", "EwiseEq"],
                   ["scalar_eq", "ScalarEq"],
                   ["ewise_log", "EwiseLog"],
                   ["ewise_exp", "EwiseExp"],
                   ["ewise_tanh", "EwiseTanh"],
                   ["reduce_max", "ReduceMax"],
                   ["reduce_sum", "ReduceSum"]]

    twoTypeName = [["from_handle", ""],
                   ["ewise_ge", "EwiseGe"],
                   ["scalar_ge", "ScalarGe"]]
    
    threeTypeName = [["ewise_add", "EwiseAdd"],
                     ["scalar_add", "ScalarAdd"],
                     ["ewise_mul", "EwiseMul"],
                     ["scalar_mul", "ScalarMul"],
                     ["ewise_div", "EwiseDiv"],
                     ["scalar_div", "ScalarDiv"],
                     # ["ewise_maximum","EwiseMaximum"],
                     # ["scalar_maximum", "ScalarMaximum"],
                     ["matmul", "Matmul"],
                     ["matmul_tiled", "MatmulTiled"]]

    codeList = (generateOneType(oneTypeName) + 
                generateTwoType(twoTypeName) + 
                generateThreeType(threeTypeName))
    
    writeTo(fileName, codeList, 4)