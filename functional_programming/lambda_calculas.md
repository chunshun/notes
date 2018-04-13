 #lambda calculas
 1. definition
`<expression>:= <name> | <function> | <application>
<function>:=lambda<name>.<expression>
<application>:=<expression><expression>`
A "name" is also called a *variable*


2. Free and bound variables
`(lambda x. x)y` :*x* is the bound variable,*y* is the bound variable
`(lambda x.x)(lambda y.yx)`: The body of the second expression is bound to the second lambda and the *x* is free.Notice that the *x* in the second expression is totally independent of the *x* in the first expression

3.Conditions
A variable `<name>` is free in a expression if one of the following three cases holds:
- `<name>` is free in `<name>`
- `<name>` is free in lambda`<name1>.<exp>` if the identifier `<name>!=<name1>` and `<name>` id free in `<exp>`
- `<name>` id free int E1E2 if `<name>` is free in E1 or if it is free in E2.
A variable `<name>` is bound if one of two cases holds:
- `<name>` id bound in lambda`<name1>.<exp>` if the identifier `<name>=<name1>` or if `<name>` is bound in `<exp>`
- `<name>` id bound in E1E2 if `<name>` is bound in E1 or if it is bound in E2

It should be emphased that the same identifier can occour free and bound in the same expression. In the expression

                (lambda x.xy)(lambda y.y)

the first *y* is free in the parenthesized subexpression to the left. It is bound in the subexpression to the right.It occurs therefore free as well as bound in the whole expression


Therefore,if the function `lambda x.<exp>` is applied to *E*,we subctitute all *free* occurences of *x* in  `<exp>` with *E*.If  the substitutio of *E* in an expression where this variable occurs bound,we rename the bound variable before performing the substiution.
`(lambda x.(lambda y).(x(lambda x.xy)))y`,we associate the argument *x* with *y*.In the body `(lambda y.(x(lambda  x.xy)))`.Before the substitution ,we have to rename the variable *y* to avoid mixing its bound with is free occurence
`[y/x](lambda t.(x(lambda x.xt))=(lambda t.(y(lambda x.xt)))`
