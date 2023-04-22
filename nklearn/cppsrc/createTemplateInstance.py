import pathlib

miniTypes = True
if miniTypes:
    intTypes = ["int32_t", "int64_t"]
    uintTypes = ["uint8_t", "uint32_t", "uint64_t"]
    floatTypes = ["_Float32", "_Float64"]
else:
    intTypes = ["int8_t", "int16_t", "int32_t", "int64_t"]
    uintTypes = ["uint8_t", "uint16_t", "uint32_t", "uint64_t"]
    floatTypes = ["_Float32", "_Float64"]
    
types = intTypes + uintTypes + floatTypes

def generateOneType(funcName):
    res = []
    # m.def("to_list", to_list<int32_t>);
    for pyFunc, cppFunc in funcName:
        if cppFunc == "":
            cppFunc = pyFunc
        for type in types:
            temp = f'm.def("{pyFunc}", {cppFunc}<{type}>);'
            res.append(temp)
    return res

def generateTwoType(funcName, firstTypes=types, SecondTypes=types):
    """两种数据类型待选
    param:
        funcName: 函数名称, 包括暴露给Python的名称c++函数名称
        firstTypes: 第一种数据类型的选择范围, 默认是全部数据类型
        secondTypes: 第二种数据类型的选择范围, 默认为全部数据类型
    """
    res = []
    # eg. m.def("from_handle", from_handle<uint64_t, uint64_t>);
    for pyFunc, cppFunc in funcName:
        if cppFunc == "":
            cppFunc = pyFunc
        for type1 in firstTypes:
            for type2 in SecondTypes:
                temp = f'm.def("{pyFunc}", {cppFunc}<{type1}, {type2}>);'
                res.append(temp)
    return res

def generateThreeType(funcName, firstTypes=types, secondTypes=types, thirdTypes=types):
    """三种数据类型待选
    param:
        funcName: 函数名称, 包括暴露给Python的名称c++函数名称
        firstTypes: 第一种数据类型的选择范围, 默认是全部数据类型
        secondTypes: 第二种数据类型的选择范围, 默认为全部数据类型
        thirdTypes: 第三种数据类型的选择范围, 默认为全部数据类型
    """
    res = []
    # eg. m.def("from_handle", from_handle<uint64_t, uint64_t>);
    for pyFunc, cppFunc in funcName:
        if cppFunc == "":
            cppFunc = pyFunc
        for type1 in firstTypes:
            for type2 in secondTypes:
                # 第三种数据类型只能为 type1 或者 type2
                if isinstance(thirdTypes, list):
                    tempTypes = thirdTypes
                else:
                    tempTypes = [type1, type2]

                for type3 in tempTypes:
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
                   
                   ["ewise_exp", "EwiseExp"],
                   
                   ["reduce_max", "ReduceMax"],
                   ["reduce_sum", "ReduceSum"]]

    twoTypeName = [["from_handle", ""],
                   ["ewise_ge", "EwiseGe"],
                   ["scalar_ge", "ScalarGe"]]
    
    # 第二种数据类型必须且只能为float
    twoTypeName_ = [["ewise_log", "EwiseLog"],
                    ["ewise_tanh", "EwiseTanh"],]
    
    threeTypeName = [["ewise_add", "EwiseAdd"],
                     ["scalar_add", "ScalarAdd"],
                     ["ewise_mul", "EwiseMul"],
                     ["scalar_mul", "ScalarMul"],
                     ["ewise_div", "EwiseDiv"],
                     ["scalar_div", "ScalarDiv"],
                     ["matmul", "Matmul"],
                     ["matmul_tiled", "MatmulTiled"]]
    
    # 第三种数据类型只能为前两种中的一种
    threeTypeName_ = [["ewise_maximum","EwiseMaximum"],
                      ["scalar_maximum", "ScalarMaximum"],]

    codeList = (generateOneType(oneTypeName) + 
                generateTwoType(twoTypeName) +
                generateTwoType(twoTypeName_, SecondTypes=floatTypes) +
                generateThreeType(threeTypeName) + 
                generateThreeType(threeTypeName_, thirdTypes="inBefor")
                )
    
    writeTo(fileName, codeList, 4)