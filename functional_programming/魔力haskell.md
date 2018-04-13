prelude模块中的类型，值和函数是默认直接可用的，在使用之前我们不需要额外的操作。然而如果需要其他模块中的一些定义，则需要使用ghci的:module方法预先加载。
表达式：
```haskell
3：：Float
```
声明：
1. 类型声明和绑定声明
Haskell中不存在变量，只存在绑定，任何时候一个名称对应的表达式都是唯一确定的。
```haskell
addone :: Int -> Int 
addone x=x+1

message::String 
message ="hello world"

```
`=`只是把右侧的表达式绑定（binding）在左侧。


代码中`x`可以看见的区域叫做作用域（scope），遵循词法结构(lexical)

```haskell
let x =1
in let y=x*2
    in x+y

```
词法作用域是指在词法结构中，被嵌套的代码段可以看到外层代码段的绑定。


2. 模块声明和导入：

haskell中没有变量和赋值，只有绑定，绑定不能改变。
haskell中没有条件，循环，分支等控制声明，条件和分支在haskell中是表达式的一部分。


函数：
1. 函数分为中缀函数和普通函数。普通函数的方法是函数跟上参数，中缀函数和算术中的加减差不多，先写第一个参数再写函数，再写参数。

```haskell
2+3
-- 中缀函数
(+) 2 3 
-- 普通函数

elem 2 [1,2,3]
2 'elem' [1,2,3]


```
2. 普通函数调用优先级最高，高于任何中缀函数
3. 中缀函数的优先级从0到9，和一致性一起定义，使用infix(不结合)，infixl(左结合)，infixr(右结合).
```haskell
2+3*4
-- 等价于2+（3×4)
```
4. -即可以在中缀函数中表示相减，也可以在普通函数中求反。两者的优先级都是6.
```haskell
2*-3
-- 报错
```
5. `：：`优先级最低，所以`::`说明的是前面整个表达式的类型。
```haskell
test 2+3 :: Double 
-- =(test 2+3):: Double 
-- \=test 2+(3::Double)
```
6. haskell的绑定名称使用驼峰命名法。
7. ghci:

        :?/:h/:help 查看帮助
        ：q  quit退出
        :l   load加载文件.hs
        :t   type查看表达式类型
        :i   info显示绑定的详细信息
8. 初级函数

data和模式匹配
数据和数据类型一直是编程过程中最核心的部分。haskell中很多控制结构和算法都建立在特定的数据结构上。
* 定义数据类型的data语法结构
* 通过模式匹配对数据进行简单的操作
* 基于data的记录语法，数据项的提取和更新

1. 数据声明data
```haskell
-- 二维坐标
data Position= MakePosition Double Double
MakePosition 1.5 2 ::Position
-- 定义了一个数据结构Position，MakePosition为构造函数


MakePosition::Double ->Double-> Position
-- MakePosition的类型

```
data关键字表示：
* 声明了一个新的类型（Position）
* 创建了该类型对应的构造函数（MakePosition)

也可以写成：
```haskell 
data  Position=Double (:+) Double 

1.5 :+  2 :: Position
(:+) 1.3 3::Position

```
中缀构造函数必须以`：`开头。
2. 模式匹配
比如计算距离：
```haskell
data Position=MakePosition Double Double 

distance :: Position -> Position-> Double 
distance p1 p2=
    case p1 of
        MakePosition x1 y1->
            case p2 of
                MakePosition x2 y2-> sqrt((x1-x2)^2+(y1-y2)^2)
```
case...of ： 模式匹配
```haskell
case x of
    pattern1 -> expression1
    pattern2 -> expression2
    ...

```
它的值取决于x满足的模式，模式匹配从上到下一次匹配pattern1到最后，一旦匹配成功，整个表达式的值就等于右侧-> 的值

    函数绑定的左侧，其实是函数名跟上模式
```haskell
distance:: Posution-> Posution-> Double

distance (MakePosition x1 y1) (MakePosition x2 y2)= sqrt((x2-x1)^2+(y2-y1)^2)
```
或者使用let...in , case遵循此法作用域， 嵌套的let内层绑定如果和外层同名，会覆盖外层绑定
```haskell
distance p1 p2=
    let MakePosition x1  y1=p1 
        MakePosition x2 y2=p2 
    in sqrt((x2-x1)^2+(y2-y1)^2) 
```
3. @pattern 
```haskell
someFunction p1@(MakePosition x1 y1) @p2(MakePosition x2 y2)=...


distance p1@(MakePosition x1 y1) p2@(MakePosition x2 y2)=
    sqrt((x1-x2)^2+(y2-y1)^2)
```
4. 多数据类型
* 多构造函数
```haskell
data Position=Cartesian Double Double | Polar Double Double
-- | 表示选择


distance (Cartesian(x1 y1))(Cartesian(x2 y2))=
    sqrt((x1-x2)^2+(y1-y2)^2)

distance(Cartesian(x2 y2))(Polar(a r))=
    let x1=r*cos a
        y1=r*sin a 
    in sqrt((x1-y1)^2+(x2-y2)^2)

distance (Polar(a r)) (Cartesian(x1 y1))=
    let x2=r*cos a
        y2=r*sin a 
    in sqrt((x1-y1)^2+(x2-y2)^2)

distance(Polar(a1 r1)) (Polar(a2 r2))=
    let x1=r1*cos a1
        y1=r1*sin a1
        x2=r2*cos a2 
        y2=r2*sin a2 
    in sqrt((x1-y1)^2+(x2-y2)^2)
```
* 完备性检查
：set -Wall / :l 1.hs

* 无参数构造函数
```haskell 
-- if ... then... else...等价于
case x of True->...
          False->...
        
```
值可以理解为不需要参数的函数
* data与类型变量
不定型以便进行抽象
```haskell
data Position a=MakePosition a a 

MakePosition 2 3 :: MakePosition Double Double
MakePosition 'a' 'e':: MakePosition Char Char

data Position=MakePosition a a 
--错误，右侧的变量必须在左侧出现过，但是相反没有这种要求

data Position a= MakePosition Int Int 
MakePosition 2 2::Position a
MakePosition 2 2::Position Char
MakePosition 2 2::Position Double 

```

* 记录语法
```haskell
getX :: Position-> Double
getX p=let MakePosition x _ =p
        in x
-- _表示占位符，只对x进行绑定
```
## 列表，递归
本质上是单链表
1. 列表
```haskell
[]:构造一个空列表，:连接一个元素和一个列表构造出一个新的列表，所有列表构造都是一个递归过程
[1,2,3]
--:[2,3]
-- 1:2:[3]
-- 1:2:3:[]
```
2. 等差序列
[..]
```haskell
[1..7]
--[1,2,3,4,5,6,7]

[1..]
--[1,2,3...]

[1,3..7]
[1,3,5,7]

[1,1..]
[1,1,1,1...]

[2,1..]
[2,1,0,-1...]

[10,9..0]
[10,9,8,7,6,5,4,3,2,1,0]

```
3. 匹配列表
对列表进行模式匹配

4. 递归操作
惰性求值 使得有限内存中放置无限长列表。               
## 元组，类型推断与高阶函数
1. 元组
data (,) a b =
2. 类型推断
3. 高阶函数
* 拉链 ，zipwith
* 柯里化

## 常用的高阶函数
1. 应用函数$与&

`$`：把左边与右边的表达式都加上括号
haskell中的管道有两个方向：`$`(从左往右)，`&`(从右往左)
2. 匿名函数(lambda)

```haskell
\pattern1 pattern2 pattern3 -> expression

\x -> x+1

```
3. 组合函数(.)
```haskell
f . g =\x-> f (g x)

```

4. 函数的补充语法

* `where`

