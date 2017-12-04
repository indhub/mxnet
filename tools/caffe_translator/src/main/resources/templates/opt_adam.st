<opt_vars(solver)>
<if(solver.momentum)>
beta1 = <solver.momentum>
<endif>
<if(solver.momentum2)>
beta2 = <solver.momentum2>
<endif>
<if(solver.delta)>
epsilon = <solver.delta>
<endif>

optimizer_params={'learning_rate':base_lr<\\>
<if(solver.wd)>, 'wd':swd<endif><\\>
<if(solver.momentum)>, 'beta1':beta1<endif><\\>
<if(solver.momentum2)>, 'beta2':beta2<endif><\\>
<if(solver.delta)>, 'epsilon':epsilon<endif>}<\\>

module.init_optimizer(optimizer='Adam', optimizer_params=optimizer_params)
